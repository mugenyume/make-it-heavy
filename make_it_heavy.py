import time
import threading
import sys
import argparse
from orchestrator import TaskOrchestrator
from providers import ProviderFactory

class OrchestratorCLI:
    def __init__(self, provider_name=None):
        self.orchestrator = TaskOrchestrator(provider_name=provider_name)
        self.start_time = None
        self.running = False
        
        # Get provider info for display
        provider_info = self.orchestrator.provider.get_provider_info()
        model_name = provider_info['model']
        
        # Clean up model name for display
        if '/' in model_name:
            model_name = model_name.split('/')[-1]
        
        model_parts = model_name.split('-')
        clean_name = '-'.join(model_parts[:3]) if len(model_parts) >= 3 else model_name
        self.model_display = f"{provider_info['display_name']} - {clean_name.upper()} HEAVY"
    
    def clear_screen(self):
        """Properly clear the entire screen"""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def format_time(self, seconds):
        """Format seconds into readable time string"""
        if seconds < 60:
            return f"{int(seconds)}S"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}M{secs}S"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}H{minutes}M"
    
    def create_progress_bar(self, status):
        """Create progress visualization based on status"""
        # ANSI color codes
        ORANGE = '\033[38;5;208m'  # Orange color
        RED = '\033[91m'           # Red color
        GREEN = '\033[92m'         # Green color
        RESET = '\033[0m'          # Reset color
        
        # Use ASCII-safe characters for Windows compatibility
        if status == "QUEUED":
            return "O " + "." * 70
        elif status == "INITIALIZING...":
            return f"{ORANGE}*{RESET} " + "." * 70
        elif status == "PROCESSING...":
            # Animated processing bar in orange
            dots = f"{ORANGE}:" * 10 + f"{RESET}" + "." * 60
            return f"{ORANGE}*{RESET} " + dots
        elif status == "COMPLETED":
            return f"{GREEN}*{RESET} " + f"{GREEN}:" * 70 + f"{RESET}"
        elif "error" in status.lower() or "failed" in status.lower():
            return f"{RED}X{RESET} " + f"{RED}x" * 70 + f"{RESET}"
        else:
            return f"{ORANGE}*{RESET} " + "." * 70
    
    def update_display(self):
        """Update the console display with current status"""
        if not self.running:
            return
            
        # Calculate elapsed time
        elapsed = time.time() - self.start_time if self.start_time else 0
        time_str = self.format_time(elapsed)
        
        # Get current progress
        progress = self.orchestrator.get_progress_status()
        
        # Clear screen properly
        self.clear_screen()
        
        # Header with dynamic model name
        print(self.model_display)
        if self.running:
            print(f"* RUNNING * {time_str}")
        else:
            print(f"* COMPLETED * {time_str}")
        print()
        
        # Agent status lines
        for i in range(self.orchestrator.num_agents):
            status = progress.get(i, "QUEUED")
            progress_bar = self.create_progress_bar(status)
            print(f"AGENT {i+1:02d}  {progress_bar}")
        
        print()
        sys.stdout.flush()
    
    def progress_monitor(self):
        """Monitor and update progress display in separate thread"""
        while self.running:
            self.update_display()
            time.sleep(0.5)  # Update every 0.5 seconds for more responsiveness
    
    def run_task(self, user_input):
        """Run orchestrator task with live progress display"""
        self.start_time = time.time()
        self.running = True
        
        # Start progress monitoring in background thread
        progress_thread = threading.Thread(target=self.progress_monitor, daemon=True)
        progress_thread.start()
        
        try:
            # Run the orchestrator
            result = self.orchestrator.orchestrate(user_input)
            
            # Stop progress monitoring
            self.running = False
            
            # Final display update
            self.update_display()
            
            # Show results
            print("=" * 80)
            print("FINAL RESULTS")
            print("=" * 80)
            print()
            print(result)
            print()
            print("=" * 80)
            
            return result
            
        except Exception as e:
            self.running = False
            self.update_display()
            print(f"\nError during orchestration: {str(e)}")
            return None
    
    def interactive_mode(self):
        """Run interactive CLI session"""
        provider_info = self.orchestrator.provider.get_provider_info()
        
        print("Multi-Agent Orchestrator")
        print(f"Provider: {provider_info['display_name']}")
        print(f"Model: {provider_info['model']}")
        print(f"Configured for {self.orchestrator.num_agents} parallel agents")
        print("Type 'quit', 'exit', or 'bye' to exit")
        print("-" * 50)
        
        try:
            print("Orchestrator initialized successfully!")
            print("Note: Make sure to set your API keys in config.yaml")
            print("-" * 50)
        except Exception as e:
            print(f"Error initializing orchestrator: {e}")
            print("Make sure you have:")
            print("1. Set your API keys in config.yaml for the selected provider")
            print("2. Installed all dependencies with: pip install -r requirements.txt")
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
                
                print("\nOrchestrator: Starting multi-agent analysis...")
                print()
                
                # Run task with live progress
                result = self.run_task(user_input)
                
                if result is None:
                    print("Task failed. Please try again.")
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                print("Please try again or type 'quit' to exit.")

def main():
    """Main entry point for the orchestrator CLI"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Multi-agent orchestrator with provider selection')
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
        # Initialize CLI with optional provider override
        cli = OrchestratorCLI(provider_name=args.provider)
        cli.interactive_mode()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Set your API keys in config.yaml for the selected provider")
        print("2. Installed all dependencies with: pip install -r requirements.txt")
        print("\nTo see available providers, run: python make_it_heavy.py --list-providers")

if __name__ == "__main__":
    main()