"""
Advanced NIE Chatbot Startup Script
Initializes all components and provides easy management
"""

import os
import sys
import time
import threading
import subprocess
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout

console = Console()

class AdvancedChatbotManager:
    def __init__(self):
        self.processes = {}
        self.status = {
            'chatbot': 'stopped',
            'monitor': 'stopped',
            'api': 'stopped'
        }
        
    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        console.print("\n🔍 Checking dependencies...")
        
        required_packages = [
            'flask', 'langchain', 'sentence-transformers', 
            'mlflow', 'prometheus-client', 'rich'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                console.print(f"✅ {package}")
            except ImportError:
                missing_packages.append(package)
                console.print(f"❌ {package}")
        
        if missing_packages:
            console.print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
            console.print("Run: pip install -r requirements.txt")
            return False
        
        console.print("✅ All dependencies satisfied!")
        return True
    
    def initialize_chatbot(self):
        """Initialize the advanced chatbot"""
        try:
            console.print("\n🤖 Initializing Advanced Chatbot...")
            
            from advanced_chatbot_rag import initialize_chatbot
            chatbot = initialize_chatbot()
            
            console.print("✅ Advanced Chatbot initialized")
            self.status['chatbot'] = 'running'
            return chatbot
            
        except Exception as e:
            console.print(f"❌ Failed to initialize chatbot: {e}")
            self.status['chatbot'] = 'error'
            return None
    
    def start_monitor(self):
        """Start MLOps monitoring"""
        try:
            console.print("\n📊 Starting MLOps Monitor...")
            
            from mlops_monitor import initialize_monitor
            monitor = initialize_monitor()
            
            console.print("✅ MLOps Monitor started")
            self.status['monitor'] = 'running'
            return monitor
            
        except Exception as e:
            console.print(f"❌ Failed to start monitor: {e}")
            self.status['monitor'] = 'error'
            return None
    
    def start_api(self):
        """Start the advanced API server"""
        try:
            console.print("\n🚀 Starting Advanced API Server...")
            
            # Start API in a separate thread
            def run_api():
                from advanced_api import app, initialize_services
                initialize_services()
                app.run(debug=False, host='0.0.0.0', port=5000)
            
            api_thread = threading.Thread(target=run_api, daemon=True)
            api_thread.start()
            
            # Wait a moment for startup
            time.sleep(3)
            
            console.print("✅ Advanced API Server started")
            console.print("🌐 Chatbot available at: http://localhost:5000")
            console.print("📊 Analytics at: http://localhost:5000/api/analytics")
            console.print("❤️ Health check at: http://localhost:5000/api/health")
            
            self.status['api'] = 'running'
            return api_thread
            
        except Exception as e:
            console.print(f"❌ Failed to start API: {e}")
            self.status['api'] = 'error'
            return None
    
    def display_status(self):
        """Display current system status"""
        table = Table(title="NIE Advanced Chatbot Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Description", style="green")
        
        status_colors = {
            'running': 'green',
            'stopped': 'red',
            'error': 'yellow'
        }
        
        table.add_row(
            "Chatbot", 
            f"[{status_colors[self.status['chatbot']]}]{self.status['chatbot']}[/]",
            "Advanced RAG with LangChain"
        )
        table.add_row(
            "Monitor", 
            f"[{status_colors[self.status['monitor']]}]{self.status['monitor']}[/]",
            "MLOps monitoring and analytics"
        )
        table.add_row(
            "API Server", 
            f"[{status_colors[self.status['api']]}]{self.status['api']}[/]",
            "Flask API with enhanced features"
        )
        
        console.print(table)
    
    def run_full_system(self):
        """Run the complete advanced chatbot system"""
        console.print(Panel.fit(
            "[bold blue]NIE Advanced Chatbot System[/bold blue]\n"
            "[green]Powered by LangChain & MLOps[/green]\n"
            f"[dim]Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
            border_style="blue"
        ))
        
        # Check dependencies
        if not self.check_dependencies():
            return False
        
        # Initialize components
        chatbot = self.initialize_chatbot()
        if not chatbot:
            return False
        
        monitor = self.start_monitor()
        if not monitor:
            console.print("⚠️ Continuing without MLOps monitoring...")
        
        api_thread = self.start_api()
        if not api_thread:
            return False
        
        # Display final status
        console.print("\n" + "="*60)
        self.display_status()
        console.print("="*60)
        
        console.print("\n🎉 [bold green]Advanced Chatbot System is running![/bold green]")
        console.print("\n[bold]Available Features:[/bold]")
        console.print("• 🔄 Dynamic response variations")
        console.print("• 🧠 Conversation memory")
        console.print("• 📊 Real-time performance monitoring")
        console.print("• 🎯 Hybrid search (semantic + keyword)")
        console.print("• 📈 User feedback collection")
        console.print("• 🔧 Auto-retraining capabilities")
        console.print("• 📱 Enhanced web interface")
        
        console.print("\n[bold]Quick Commands:[/bold]")
        console.print("• Press Ctrl+C to stop the system")
        console.print("• Visit http://localhost:5000 for the chatbot")
        console.print("• Check analytics at http://localhost:5000/api/analytics")
        
        try:
            # Keep the main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n\n🛑 Shutting down Advanced Chatbot System...")
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        console.print("🧹 Cleaning up resources...")
        
        if self.status['monitor'] == 'running':
            try:
                from mlops_monitor import get_monitor
                monitor = get_monitor()
                if monitor:
                    monitor.stop_monitoring()
                    console.print("✅ Monitor stopped")
            except:
                pass
        
        console.print("✅ Cleanup completed")

def main():
    """Main entry point"""
    manager = AdvancedChatbotManager()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'test':
            console.print("🧪 Testing system components...")
            if manager.check_dependencies():
                chatbot = manager.initialize_chatbot()
                if chatbot:
                    console.print("✅ System test passed!")
                else:
                    console.print("❌ System test failed!")
        elif command == 'status':
            manager.display_status()
        else:
            console.print(f"Unknown command: {command}")
            console.print("Available commands: test, status")
    else:
        # Run full system
        manager.run_full_system()

if __name__ == "__main__":
    main()
