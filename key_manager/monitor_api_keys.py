"""
Real-time API Key Usage Monitor
Shows live statistics of API key usage during batch processing.
"""

import time
import os
import sys
from datetime import datetime

# Add parent directory to Python path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from .api_key_manager import get_key_usage_stats, get_api_key_manager
except ImportError:
    # If relative import fails (when running directly), use absolute import
    from key_manager.api_key_manager import get_key_usage_stats, get_api_key_manager

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def format_time(timestamp_str):
    """Format timestamp for display."""
    if not timestamp_str:
        return "Never"
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime("%H:%M:%S")
    except:
        return "Invalid"

def display_usage_stats():
    """Display current usage statistics."""
    try:
        manager = get_api_key_manager()
        stats = get_key_usage_stats()
        
        print("🔑 API Key Manager - Live Usage Monitor")
        print("=" * 60)
        print(f"⏰ Last updated: {datetime.now().strftime('%H:%M:%S')}")
        print(f"📊 Rate limit: {manager.rate_limit_per_minute} requests per minute per key")
        print(f"🔄 Current active key: key_{manager.current_key_index + 1}")
        print()
        
        # Header
        print(f"{'Key':<8} {'Current':<10} {'Total':<8} {'Status':<12} {'Last Used':<10} {'Reset Time':<10}")
        print("-" * 60)
        
        for key_id, stat in stats.items():
            current_usage = f"{stat['requests_made_current_window']}/{manager.rate_limit_per_minute}"
            total_requests = stat['total_requests']
            
            if stat['is_blocked']:
                status = "🚫 BLOCKED"
                reset_time = format_time(stat['rate_limit_reset'])
            else:
                status = "✅ Available"
                reset_time = "-"
            
            last_used = format_time(stat['last_used'])
            
            print(f"{key_id:<8} {current_usage:<10} {total_requests:<8} {status:<12} {last_used:<10} {reset_time:<10}")
        
        # Summary
        total_requests = sum(stat['total_requests'] for stat in stats.values())
        blocked_keys = sum(1 for stat in stats.values() if stat['is_blocked'])
        available_keys = len(stats) - blocked_keys
        
        print()
        print(f"📈 Summary:")
        print(f"   Total requests made: {total_requests}")
        print(f"   Available keys: {available_keys}/{len(stats)}")
        print(f"   Blocked keys: {blocked_keys}/{len(stats)}")
        
        # Calculate current throughput
        current_capacity = available_keys * manager.rate_limit_per_minute
        max_capacity = len(stats) * manager.rate_limit_per_minute
        print(f"   Current capacity: {current_capacity}/{max_capacity} req/min")
        
        return True
        
    except Exception as e:
        print(f"❌ Error getting usage stats: {e}")
        return False

def monitor_usage(refresh_interval=2):
    """Monitor usage with auto-refresh."""
    print("🔄 Starting real-time monitor...")
    print("📝 Press Ctrl+C to stop monitoring")
    print()
    
    try:
        while True:
            clear_screen()
            if not display_usage_stats():
                break
            
            print(f"\n🔄 Refreshing every {refresh_interval} seconds... (Ctrl+C to stop)")
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped.")
    except Exception as e:
        print(f"\n❌ Monitor error: {e}")

def show_one_time_stats():
    """Show stats once without monitoring."""
    display_usage_stats()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # Show stats once
        show_one_time_stats()
    else:
        # Start monitoring
        refresh_interval = 2
        if len(sys.argv) > 1:
            try:
                refresh_interval = int(sys.argv[1])
            except ValueError:
                print("Usage: python monitor_api_keys.py [refresh_interval_seconds]")
                print("   or: python monitor_api_keys.py --once")
                sys.exit(1)
        
        monitor_usage(refresh_interval)
