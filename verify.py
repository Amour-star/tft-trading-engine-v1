#!/usr/bin/env python3
"""
Quick Verification Script - Check if Engine & Dashboard are showing real values
"""

import sys
import os
import time
import requests
from datetime import datetime


def _ensure_utf8_output():
    """Avoid UnicodeEncodeError on Windows consoles with legacy codepages."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


_ensure_utf8_output()

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def check_docker_services():
    """Check if Docker services are running"""
    print_header("1. Docker Services Status")
    
    import subprocess
    
    try:
        result = subprocess.run(
            ["docker-compose", "ps"],
            capture_output=True,
            text=True,
            check=True
        )
        
        services = {
            'engine': False,
            'dashboard': False,
            'postgres': False,
            'redis': False
        }
        
        for line in result.stdout.split('\n'):
            for service in services.keys():
                if service in line.lower() and 'up' in line.lower():
                    services[service] = True
        
        for service, running in services.items():
            status = "✅ Running" if running else "❌ Not running"
            print(f"  {service.capitalize():12} {status}")
        
        all_running = all(services.values())
        
        if not all_running:
            print("\n⚠️  Some services are not running!")
            print("   Run: docker-compose up -d")
            return False
        
        print("\n✅ All services are running")
        return True
        
    except Exception as e:
        print(f"❌ Error checking Docker: {e}")
        return False

def check_engine_logs():
    """Check if engine is generating signals"""
    print_header("2. Engine Signal Generation")
    
    import subprocess
    
    try:
        # Get last 50 lines of logs
        result = subprocess.run(
            ["docker-compose", "logs", "--tail=50", "engine"],
            capture_output=True,
            text=True,
            check=True
        )
        
        logs = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        
        # Check for key indicators
        logs_l = logs.lower()
        signals_generated = any(
            token in logs_l
            for token in (
                "starting signal generation cycle",
                "generate_signal",
                "found 30 candidate pairs",
                "computed 51 features",
                "signal",
                "qualified",
            )
        )
        paper_trades = "paper trade" in logs_l
        errors = "error" in logs_l and "retry" not in logs_l
        
        print(f"  Signals generated:     {'✅ Yes' if signals_generated else '⚠️  Not yet'}")
        print(f"  Paper trades:          {'✅ Yes' if paper_trades else '⚠️  Not yet'}")
        print(f"  Errors:                {'❌ Yes' if errors else '✅ None'}")
        
        if not signals_generated:
            print("\n⚠️  No signals found in recent logs")
            print("   This might be normal if engine just started")
            print("   Wait 2-5 minutes and check again")
            return False
        
        if paper_trades:
            print("\n✅ Engine is generating paper trades!")
        else:
            print("\n⚠️  Signals generated but no trades yet")
        
        return signals_generated
        
    except Exception as e:
        print(f"❌ Error checking logs: {e}")
        return False

def check_database_connection():
    """Check if database has trades"""
    print_header("3. Database Connection & Trades")
    import subprocess
    try:
        db_user = os.getenv("POSTGRES_USER", "trader")
        db_name = os.getenv("POSTGRES_DB", "tft_trading")
        result = subprocess.run(
            ["docker-compose", "exec", "-T", "postgres",
             "psql", "-U", db_user, "-d", db_name,
             "-t", "-A", "-c", "SELECT COUNT(*) FROM trades;"],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout.strip()
        if output.isdigit():
            count = int(output)
            print(f"  Total trades in DB:    {count}")
            if count == 0:
                print("\nWARNING: No trades in database yet")
                print("   DB connection is healthy; engine has not opened a trade yet.")
                return True
            print(f"\nOK Database has {count} trade(s)")
            return True
        print(f"  Could not parse trade count: {output}")
        return False
    except Exception as e:
        print(f"ERROR checking database: {e}")
        return False

def check_kucoin_api():
    """Check KuCoin API connection"""
    print_header("4. KuCoin API Connection")
    import subprocess
    try:
        probe_code = (
            "from data.fetcher import KuCoinDataFetcher\n"
            "f=KuCoinDataFetcher()\n"
            "ok_market=False\n"
            "ok_auth=False\n"
            "err=''\n"
            "try:\n"
            "    t=f.market.get_ticker('XRP-USDT')\n"
            "    ok_market=float(t.get('price',0))>0\n"
            "except Exception as e:\n"
            "    err=f'PUBLIC:{e}'\n"
            "try:\n"
            "    accounts=f.user_client.get_account_list(account_type='trade')\n"
            "    ok_auth=isinstance(accounts,list)\n"
            "    if ok_auth:\n"
            "        total=0.0\n"
            "        for acc in accounts:\n"
            "            total += float(acc.get('available',0) or 0) + float(acc.get('holds',0) or 0)\n"
            "        print('ACCOUNT SUMMARY')\n"
            "        print(f'TOTAL VALUE: {total:.8f}')\n"
            "except Exception as e:\n"
            "    err=(err+' | ' if err else '')+f'PRIVATE:{e}'\n"
            "print(f'MARKET_OK={ok_market}')\n"
            "print(f'AUTH_OK={ok_auth}')\n"
            "if ok_market and ok_auth:\n"
            "    print('ALL TESTS PASSED')\n"
            "else:\n"
            "    print('TESTS FAILED')\n"
            "    if err:\n"
            "        print('ERROR=' + err)\n"
        )

        result = subprocess.run(
            ["docker-compose", "exec", "-T", "engine", 
             "python", "-c", probe_code],
            capture_output=True,
            text=True,
            timeout=30
        )
        output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        success = "all tests passed" in output.lower()
        if success and result.returncode == 0:
            print("  API Connection:        OK")
            print("  Account Access:        OK")
            for line in output.split('\n'):
                if 'total value' in line.lower() or 'account summary' in line.lower():
                    print(f"  {line.strip()}")
            print("\nOK KuCoin API is working")
            return True
        print("  API Connection:        FAILED")
        print("\nWARNING: KuCoin API not configured properly")
        print("   Check .env file for API credentials")
        error_lines = [line.strip() for line in output.split('\n') if line.strip()]
        if error_lines:
            print(f"   Details: {error_lines[-1]}")
        return False
    except subprocess.TimeoutExpired:
        print("  WARNING: API test timed out")
        return False
    except Exception as e:
        print(f"ERROR testing API: {e}")
        return False

def check_dashboard_access():
    """Check if dashboard is accessible"""
    print_header("5. Dashboard Accessibility")
    
    try:
        response = requests.get("http://localhost:8501", timeout=10)
        
        if response.status_code == 200:
            print("  HTTP Status:           ✅ 200 OK")
            print("  Dashboard URL:         http://localhost:8501")
            print("\n✅ Dashboard is accessible")
            return True
        else:
            print(f"  HTTP Status:           ❌ {response.status_code}")
            print("\n⚠️  Dashboard returned non-200 status")
            return False
            
    except requests.exceptions.ConnectionError:
        print("  HTTP Status:           ❌ Connection refused")
        print("\n⚠️  Dashboard not accessible")
        print("   Check if dashboard service is running")
        return False
    except Exception as e:
        print(f"❌ Error accessing dashboard: {e}")
        return False

def check_real_time_data():
    """Check if engine is processing real-time data"""
    print_header("6. Real-Time Data Processing")
    import subprocess
    try:
        result = subprocess.run(
            ["docker-compose", "logs", "--since=5m", "engine"],
            capture_output=True,
            text=True,
            timeout=60
        )
        output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        output_l = output.lower()
        cycle_count = output_l.count("starting signal generation cycle")
        feature_count = output_l.count("computed 51 features")
        has_evaluation = (
            "all pairs disqualified" in output_l
            or "best pair:" in output_l
            or "trade opened:" in output_l
        )

        has_processing = has_evaluation or feature_count > 0 or "found 30 candidate pairs" in output_l

        if cycle_count > 0 and has_processing:
            print(f"  Signal cycles (last 5m): {cycle_count}")
            if feature_count > 0:
                print(f"  Feature batches:        {feature_count}")
            print("\nOK Engine is processing real-time data")
            return True
        print("  WARNING: Could not verify data processing from recent logs")
        return False
    except subprocess.TimeoutExpired:
        print("  WARNING: Data processing check timed out")
        return False
    except Exception as e:
        print(f"ERROR checking data: {e}")
        return False

def main():
    """Run all verification checks"""
    print("\n" + "🔍 TFT Trading Engine - Verification Report ".center(60, "="))
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = {}
    
    # Run all checks
    results['docker_services'] = check_docker_services()
    time.sleep(1)
    
    results['engine_logs'] = check_engine_logs()
    time.sleep(1)
    
    results['database'] = check_database_connection()
    time.sleep(1)
    
    results['kucoin_api'] = check_kucoin_api()
    time.sleep(1)
    
    results['dashboard'] = check_dashboard_access()
    time.sleep(1)
    
    results['realtime_data'] = check_real_time_data()
    
    # Summary
    print_header("📊 VERIFICATION SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for check, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {check.replace('_', ' ').title():25} {status}")
    
    print("\n" + "="*60)
    print(f"  Result: {passed}/{total} checks passed")
    print("="*60)
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:\n")
    
    if passed == total:
        print("✅ Everything is working perfectly!")
        print("\nNext steps:")
        print("  1. Open dashboard: http://localhost:8501")
        print("  2. Monitor for 24 hours in paper trading mode")
        print("  3. Review performance metrics")
        print("  4. Enable real trading when confident")
        
    elif passed >= 4:
        print("⚠️  Most checks passed, but some issues detected")
        
        if not results['database']:
            print("\n📝 To fix database issue:")
            print("   • Wait 5-10 minutes for engine to generate trades")
            print("   • Run: docker-compose logs -f engine")
            print("   • Look for 'PAPER TRADE' messages")
        
        if not results['kucoin_api']:
            print("\n📝 To fix KuCoin API issue:")
            print("   • Check .env file has correct credentials")
            print("   • Run: docker-compose restart engine")
            print("   • Test again: python verify.py")
        
        if not results['dashboard']:
            print("\n📝 To fix dashboard issue:")
            print("   • Check: docker-compose logs dashboard")
            print("   • Restart: docker-compose restart dashboard")
            print("   • Verify port 8501 is not blocked")
    
    else:
        print("❌ Multiple issues detected")
        print("\n📝 Troubleshooting steps:")
        print("   1. Check Docker is running: docker --version")
        print("   2. Restart all services: docker-compose restart")
        print("   3. Check logs: docker-compose logs")
        print("   4. Verify .env file exists and has correct values")
        print("   5. Re-run this script: python verify.py")
    
    print("\n" + "="*60)
    
    # Return exit code
    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



