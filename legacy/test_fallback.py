"""
Test script for the fallback system
"""

from fallback_system import get_fallback_response, initialize_fallback
import time

def test_fallback_system():
    """Test the fallback system with various queries"""
    print("🧪 Testing NIE Website Fallback System")
    print("=" * 50)
    
    # Initialize fallback
    initialize_fallback()
    
    # Test queries that should trigger fallback
    test_queries = [
        "latest admission criteria 2024",
        "new courses offered this year", 
        "recent placement statistics",
        "hostel rules and regulations",
        "library timings and facilities",
        "something completely unrelated to NIE"  # Should get generic fallback
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {query}")
        print("-" * 30)
        
        start_time = time.time()
        response, confidence = get_fallback_response(query)
        response_time = time.time() - start_time
        
        print(f"⏱️ Response Time: {response_time:.2f}s")
        print(f"🎯 Confidence: {confidence:.2f}")
        print(f"📝 Response: {response[:200]}...")
        
        if confidence >= 0.4:
            print("✅ External fallback successful")
        else:
            print("⚠️ Using generic fallback")
        
        time.sleep(1)  # Be respectful to the server
    
    print("\n🎉 Fallback system testing completed!")

if __name__ == "__main__":
    test_fallback_system()
