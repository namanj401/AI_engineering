"""
Synthetic Log Parser Dataset Generator for Fine-Tuning LLMs
Outputs chat-templated training examples (Llama-3 Instruct format) for fine-tuning a model to parse unformatted raw logs into strict structured JSON.
"""

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# -----------------------------------------------------------------------------
# 1. System Prompt Definition
# -----------------------------------------------------------------------------
# Strict system prompt designed to prevent conversational drift / pleasantries.
SYSTEM_PROMPT = (
    "You are an automated backend log-parsing engine. Your only task is to convert raw log entries into a strictly valid JSON object adhering to the schema.\n\n"
    "CRITICAL RULES:\n"
    "1. Output MUST start with '{' and end with '}'.\n"
    "2. DO NOT output conversational text, explanations, or notes before or after the JSON.\n"
    "3. DO NOT wrap the output in markdown codeblocks (e.g. ```json or ```).\n"
    "4. All keys must be present. If an optional field is missing, use null.\n"
    "5. Output valid, parseable JSON only."
)

# -----------------------------------------------------------------------------
# 2. Template Elements for Generating Realistic Dummy Logs
# -----------------------------------------------------------------------------
SERVICES = [
    "auth-service",
    "payment-gateway",
    "order-service",
    "database-cluster",
    "api-gateway",
    "inventory-worker",
    "notification-dispatcher",
]

SEVERITIES = ["INFO", "WARNING", "ERROR", "CRITICAL"]

ERROR_SCENARIOS = [
    {
        "service": "auth-service",
        "severity": "ERROR",
        "code": "ERR_AUTH_EXPIRED",
        "templates": [
            "Session validation failed: JWT signature expired for user_id={user_id}",
            "OAuth2 token refresh rejected for user_id={user_id} due to expiration",
        ],
    },
    {
        "service": "payment-gateway",
        "severity": "CRITICAL",
        "code": "ERR_PAYMENT_TIMEOUT",
        "templates": [
            "Stripe API timeout after 30000ms for transaction_id={txn_id}",
            "Downstream merchant banking gateway unreachable: connection reset by peer on txn {txn_id}",
        ],
    },
    {
        "service": "database-cluster",
        "severity": "ERROR",
        "code": "ERR_DB_LOCK_DEADLOCK",
        "templates": [
            "Deadlock detected on table 'orders' during transaction {txn_id}; rolling back",
            "PostgreSQL connection pool exhausted: active_connections=500, waiting_queries=42",
        ],
    },
    {
        "service": "api-gateway",
        "severity": "WARNING",
        "code": "ERR_RATE_LIMIT_429",
        "templates": [
            "Client IP {ip} exceeded quota (120 req/sec) on endpoint /api/v2/checkout",
            "Rate limiter tripped for tenant_id={user_id}; returning HTTP 429 Too Many Requests",
        ],
    },
    {
        "service": "inventory-worker",
        "severity": "INFO",
        "code": None,
        "templates": [
            "Successfully synced {count} SKU items from warehouse batch #{txn_id}",
            "Periodic cache invalidation completed in {ms}ms across all nodes",
        ],
    },
    {
        "service": "notification-dispatcher",
        "severity": "WARNING",
        "code": "ERR_EMAIL_BOUNCE",
        "templates": [
            "SMTP server rejected message to recipient@{domain}: 550 Mailbox unavailable",
            "Twilio SMS delivery failed for destination +1-555-01{ip_octet}: carrier rejected message",
        ],
    },
]

LOG_FORMAT_STYLES = [
    # Style 1: Standard Syslog-like
    "{timestamp} [{severity}] {service}: [{error_code}] {message}",
    # Style 2: Microservice bracketed
    "[{timestamp}] [{service}] [{severity}] ({error_code}) - {message}",
    # Style 3: Key-value log
    'time="{timestamp}" level={severity} service="{service}" code="{error_code}" msg="{message}"',
    # Style 4: Plain text production log
    "{timestamp} {service} - {severity} - {error_code} - {message}",
]


def generate_single_log() -> Dict[str, Any]:
    """Generates a single synthetic raw log and its corresponding ground-truth JSON."""
    scenario = random.choice(ERROR_SCENARIOS)
    service = scenario["service"]
    severity = scenario["severity"]
    error_code = scenario["code"]

    # Generate realistic dynamic values
    user_id = random.randint(1000, 99999)
    txn_id = f"txn_{random.randint(100000, 999999)}"
    ip = f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}"
    ip_octet = random.randint(10, 99)
    count = random.randint(50, 1500)
    ms = random.randint(10, 850)
    domain = random.choice(["example.com", "corp.net", "mailservice.io"])

    raw_template = random.choice(scenario["templates"])
    message = raw_template.format(
        user_id=user_id,
        txn_id=txn_id,
        ip=ip,
        ip_octet=ip_octet,
        count=count,
        ms=ms,
        domain=domain,
    )

    # ISO-8601 Timestamp
    now = datetime.now(timezone.utc)
    timestamp_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Define the exact ground-truth JSON schema
    target_json = {
        "timestamp": timestamp_str,
        "service_name": service,
        "severity": severity,
        "error_code": error_code,
        "message": message,
    }

    # Format the raw input log string
    style = random.choice(LOG_FORMAT_STYLES)
    raw_log = style.format(
        timestamp=timestamp_str,
        severity=severity,
        service=service,
        error_code=error_code if error_code else "NONE",
        message=message,
    )

    return {"raw_log": raw_log, "target_json": target_json}


# -----------------------------------------------------------------------------
# 3. Chat / Instruct Template Formatter (Llama-3 Format)
# -----------------------------------------------------------------------------
def format_llama3_chat(system_prompt: str, user_prompt: str, assistant_response: str) -> str:
    """
    Wraps the conversation into the official Llama-3 Instruct special tokens:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {system}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    {assistant}<|eot_id|>
    """
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"Parse the following raw log into structured JSON:\n{user_prompt}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{assistant_response}<|eot_id|>"
    )


# -----------------------------------------------------------------------------
# 4. Dataset Generation Pipeline
# -----------------------------------------------------------------------------
def generate_log_dataset(num_samples: int = 50) -> List[Dict[str, Any]]:
    """
    Generates a list of training examples formatted for causal language model fine-tuning.
    """
    dataset = []

    for _ in range(num_samples):
        sample = generate_single_log()
        raw_log = sample["raw_log"]
        # Compact, deterministic JSON string representation
        target_json_str = json.dumps(sample["target_json"], ensure_ascii=False)

        # Full training text formatted with Llama-3 special tokens
        formatted_training_text = format_llama3_chat(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=raw_log,
            assistant_response=target_json_str,
        )

        dataset.append({
            "text": formatted_training_text,
            "raw_log": raw_log,
            "target_json": sample["target_json"],
        })

    return dataset


if __name__ == "__main__":
    # Generate dataset
    dataset = generate_log_dataset(num_samples=100)

    print("=" * 80)
    print("SINGLE COMPLETED TRAINING EXAMPLE (RAW STRING SEEN BY THE MODEL):")
    print("=" * 80)
    # Print the exact formatted text of the first training sample
    print(dataset[0]["text"])
    print("=" * 80)

    # Optional: Save to JSONL for Hugging Face / Axolotl / Unsloth training
    output_file = Path(__file__).resolve().parent / "log_parser_dataset.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps({"text": entry["text"]}) + "\n")

    print(f"\nGenerated {len(dataset)} examples and saved to: {output_file.name}")
