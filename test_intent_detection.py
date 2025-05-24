"""
Test intent detection for user's specific queries
"""
import sys
sys.path.append('.')

from app.processors.smart_query_processor import SmartQueryProcessor

def test_intent_detection():
    """Test intent detection for specific queries"""
    
    print("🔍 Testing Intent Detection")
    print("=" * 50)
    
    processor = SmartQueryProcessor()
    
    test_queries = [
        "what can you explain about both files logical relations?",
        "can you find out how many transactions missing (the count) and how much is missing in RM?",
        "Based on this information that Payment by order is the total payment received to the 10 camp enterprise shop from different transaction type and MMSDO is only showing the payment to that shop via QRpay, can you find out how many transactions missing (the count) and how much is missing in RM?"
    ]
    
    available_files = [
        "temp/MMSDO_P_202412_EP810177.csv",
        "temp/Payments by order - 2024-12-01 - 2024-12-31.csv"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query[:60]}...'")
        
        intent = processor.detect_intent(query, available_files)
        
        print(f"   Intent: {intent.intent_type}")
        print(f"   Confidence: {intent.confidence:.2f}")
        print(f"   Parameters: {intent.parameters}")
        
        # Show which patterns matched
        query_lower = query.lower()
        print(f"   Pattern matches:")
        
        for intent_type, patterns in processor.intent_patterns.items():
            matches = []
            for pattern in patterns:
                import re
                if re.search(pattern, query_lower):
                    matches.append(pattern)
            
            if matches:
                print(f"     {intent_type}: {matches}")

if __name__ == "__main__":
    test_intent_detection() 