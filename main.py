import argparse
import sys
from agent import AIAgent
from providers import ProviderFactory

def main():
    """Main entry point for the AI agent with provider selection"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AI Agent with multi-provider support')
    parser.add_argument('--provider', choices=ProviderFactory.get_available_providers(),
                        help='AI provider to use (overrides config.yaml)')
    parser.add_argument('--list-providers', action='store_true',
                        help='List available providers and exit')
    
    args = parser.parse_args()
    
    # List providers if requested
    if args.list_providers:
        print("Available AI Providers:")
        print("-" * 30)
        for provider_name in ProviderFactory.get_available_providers():
            info = ProviderFactory.get_provider_info(provider_name)
            print(f"• {info['display_name']} ({provider_name})")
            print(f"  Description: {info['description']}")
            print(f"  Default model: {info['default_model']}")
            print()
        return
    
    try:
        # Initialize agent with optional provider override
        agent = AIAgent(provider_name=args.provider)
        
        provider_info = agent.provider_info
        print("Multi-Provider AI Agent")
        print("-" * 50)
        print(f"Provider: {provider_info['display_name']}")
        print(f"Model: {provider_info['model']}")
        print("Type 'quit', 'exit', or 'bye' to exit")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error initializing agent: {e}")
        print("\nMake sure you have:")
        print("1. Set your API keys in config.yaml for the selected provider")
        print("2. Installed all dependencies with: pip install -r requirements.txt")
        print("\nTo see available providers, run: python main.py --list-providers")
        return
    
    while True:
        try:
            user_input = input("\nUser: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
            
            if not user_input:
                print("Please enter a question or command.")
                continue
            
            print("Agent: Thinking...")
            response = agent.run(user_input)
            print(f"Agent: {response}")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again or type 'quit' to exit.")

if __name__ == "__main__":
    main()