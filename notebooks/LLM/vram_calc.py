import string
from argparse import ArgumentParser
import time
import ollama
def calculate_vram(params_billions,bits,overhead = 1.2):
    return params_billions * bits * overhead / 8

def run_inference(model_name,prompt=None):
    start_time = time.perf_counter()
    response = None
    
    if prompt is not None:
        response = ollama.generate(model_name,prompt=prompt)
    
    end_time = time.perf_counter()

    duration = end_time-start_time
    return response,duration

if __name__ == "__main__":
    parser = ArgumentParser(description= "Estimate Vram required for load and run an LLM")
    parser.add_argument(
        "--params","-p",
        type = float,
        required= True ,
        help= "Number of model parameters in billions")
    
    parser.add_argument("--bits","-b",
    type = float,
    required = True,
    help = "Precision in bits for each parameter")

    parser.add_argument("--overhead","-o",
    type = float,
    default = 1.2,
    help = "Overhead multiplier")

    parser.add_argument("--model","-m",
    type = str,
    default="llama3",
    help="Model name to be used by the ollama")

    parser.add_argument("--prompt","-pr",
    type=str,
    help="Prompt to be given to the model")

    args = parser.parse_args()

    vram_gb = calculate_vram(args.params,args.bits,args.overhead)
    print(f"Total number of parameters: {args.params:.2f} billion")
    print(f"Bit precision: {args.bits:.2f} bits")
    print(f"Estimated VRAM Required: {vram_gb:.2f} GB"  )


    response,duration = run_inference(args.model,args.prompt)
    if(response !=None):
        throughput = response['eval_count'] * 1e9 / response['eval_duration']
        print(f"Total number of tokens generated: {response['eval_count']}")
        print(f"Total time taken: {response['eval_duration'] * 1e-9 :.2f} seconds")
        print(f"Token throughput:{throughput:.2f}\n\n\n\n")
        print(response['response'])