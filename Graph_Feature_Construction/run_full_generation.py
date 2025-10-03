import sys
import os
import argparse


from generate_content_embeddings import ContentEmbeddingGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate full node content and embedding")
    parser.add_argument('--repo_name', type=str, default='astropy', help='name of repo')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--output_dir', type=str, default='./output', help='output_dir')
    parser.add_argument('--max_nodes', type=int, default=None, help='max nodes to be test')
    
    args = parser.parse_args()
    
    print(f"Start to generate repo {args.repo_name} 's full node content and embedding")
    print(f"args:")
    print(f"repo_name: {args.repo_name}")
    print(f"batch_size: {args.batch_size}")
    print(f"output_dir: {args.output_dir}")
    print(f"max_nodes: {args.max_nodes or 'all'}")
    
    
    generator = ContentEmbeddingGenerator(
        repo_name=args.repo_name,
        output_dir=args.output_dir
    )
    
    try:
        
        node_ids = generator.load_graph_data()
        
        
        if args.max_nodes and args.max_nodes < len(node_ids):
            node_ids = node_ids[:args.max_nodes]
            print(f"limit the number of nodes to: {len(node_ids)}")
        
        
        dataset = generator.load_dataset()
        
        
        node_contents = generator.extract_node_contents(node_ids, dataset)
        
        
        valid_nodes = {k: v for k, v in node_contents.items() 
                      if v['attr'] and v['attr'] != '{}'}
        print(f"Node Statistics:")
        print(f"number of all nodes: {len(node_contents)}")
        print(f"Valid nodes(with content): {len(valid_nodes)}")
        print(f"Empty nodes: {len(node_contents) - len(valid_nodes)}")
        
        
        #generator.save_content_json(node_contents)
        
       
        print(f"Start to generate {len(node_contents)} nodes's embedding (including empty nodes)...")
        embeddings = generator.generate_embeddings(node_contents, batch_size=args.batch_size)
        
       
        if embeddings:
            generator.save_embeddings_safetensor(embeddings)
        else:
            print("Valid node content not found, skip generation")
        
        print(f"Finished！")
        
    except Exception as e:
        print(f"Failured: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
