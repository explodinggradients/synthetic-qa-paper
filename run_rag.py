# todo: ship the conda env along with the codes
# todo: time each part of the pipeline to find out potential places for optimization

from rag_utils import VectorBase, Retriever, Generator
import argparse, yaml, json, torch


DTYPE_2_TORCH_DTYPE = {
    'bf16': torch.bfloat16,
    'fp16': torch.float16,
    'fp32': torch.float32
}


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="This script accepts two command-line arguments.")
    parser.add_argument('--yaml-config', dest='yaml_config', required=True)
    parser.add_argument('--mode', dest='mode', type=str) 
    args = parser.parse_args()
    
    # Load configuration from a YAML file
    with open(args.yaml_config, 'r') as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)
        config['generator_model_torch_dtype'] = DTYPE_2_TORCH_DTYPE[config['generator_model_torch_dtype']]
    
    if 'v' in args.mode:
        vb = VectorBase.from_yaml_config(config)
    
    if 'r' in args.mode:
        ret = Retriever.from_yaml_config(vb, config)
        # print(ret.ls_rets[list(ret.ls_rets.keys())[0]][0])
        
    if 'g' in args.mode:
        gen = Generator.from_yaml_config(ret, config)



