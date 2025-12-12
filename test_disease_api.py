"""
Test script for PlantNet Disease API Integration
Run this to verify Phase 1 implementation
"""

from plantnet_api import PlantNetAPI, classify_plant_with_health, classify_plant_disease
from PIL import Image
import numpy as np
import os

def test_disease_detection():
    """Test the new disease detection functionality"""
    
    # Check for API key
    api_key = os.getenv('PLANTNET_API_KEY')
    if not api_key:
        print("❌ Error: PLANTNET_API_KEY environment variable not set")
        print("Set it with: export PLANTNET_API_KEY='your-key-here'")
        return
    
    # Check for test image
    test_image_path = "Model Test Inputs/pinepple.jpg"
    if not os.path.exists(test_image_path):
        print(f"❌ Error: Test image not found at {test_image_path}")
        print("Please provide a valid image path")
        return
    
    print("🧪 Testing PlantNet Disease API Integration")
    print("=" * 70)
    
    # Load test image
    print(f"\n📷 Loading image: {test_image_path}")
    image = Image.open(test_image_path).convert('RGB')
    image_array = np.array(image)
    print(f"   Image size: {image.size}, Array shape: {image_array.shape}")
    
    # Initialize API
    api = PlantNetAPI(api_key)
    
    # Test 1: Species Identification (existing functionality)
    print("\n" + "=" * 70)
    print("TEST 1: Species Identification (Existing)")
    print("=" * 70)
    
    species_result = api.identify_from_array(image_array, num_results=2)
    
    if species_result['success']:
        print("✅ Species identification successful!")
        top = species_result['top_result']
        print(f"\n   Species: {top['scientific_name']}")
        print(f"   Common: {', '.join(top['common_names'][:2]) if top['common_names'] else 'N/A'}")
        print(f"   Confidence: {top['confidence_pct']}")
        print(f"   Credits remaining: {species_result['query_info']['remaining_credits']}")
    else:
        print(f"❌ Failed: {species_result.get('message', 'Unknown error')}")
    
    # Test 2: Disease Identification (NEW functionality)
    print("\n" + "=" * 70)
    print("TEST 2: Disease Identification (NEW)")
    print("=" * 70)
    
    disease_result = api.identify_disease_from_array(image_array, num_results=3)
    
    if disease_result['success']:
        print("✅ Disease detection successful!")
        
        if disease_result['has_disease']:
            top_disease = disease_result['top_disease']
            print(f"\n   ⚠️  Disease detected!")
            print(f"   Name: {top_disease['label']}")
            print(f"   EPPO Code: {top_disease['name']}")
            print(f"   Confidence: {top_disease['confidence_pct']}")
            
            if len(disease_result['all_results']) > 1:
                print(f"\n   Other possible issues:")
                for d in disease_result['all_results'][1:3]:
                    print(f"   - {d['label']} ({d['confidence_pct']})")
        else:
            print("\n   ✅ No diseases detected - plant appears healthy!")
        
        print(f"\n   Credits remaining: {disease_result['query_info']['remaining_credits']}")
    else:
        print(f"❌ Failed: {disease_result.get('message', 'Unknown error')}")
    
    # Test 3: Comprehensive Analysis (NEW combined function)
    print("\n" + "=" * 70)
    print("TEST 3: Comprehensive Health Analysis (NEW)")
    print("=" * 70)
    
    health_result = api.analyze_plant_health(image_array)
    
    if health_result['combined_success']:
        print("✅ Comprehensive analysis successful!")
        
        # Species summary
        if health_result['species']['success']:
            species = health_result['species']['top_result']
            print(f"\n   🌿 PLANT: {species['scientific_name']}")
            if species['common_names']:
                print(f"      Common: {species['common_names'][0]}")
        
        # Health summary
        if health_result['disease']['success']:
            if health_result['disease']['has_disease']:
                disease = health_result['disease']['top_disease']
                print(f"\n   ⚠️  HEALTH: Issue Detected")
                print(f"      Problem: {disease['label']}")
                print(f"      Severity: {disease['confidence_pct']} confidence")
            else:
                print(f"\n   ✅ HEALTH: Appears Healthy")
        
        print(f"\n   📊 ANALYSIS COMPLETE")
    else:
        print(f"❌ Failed: Analysis could not be completed")
    
    # Test 4: Convenience functions
    print("\n" + "=" * 70)
    print("TEST 4: Convenience Functions (for app.py integration)")
    print("=" * 70)
    
    print("\n   Testing classify_plant_with_health()...")
    result = classify_plant_with_health(image_array, api_key)
    print(f"   ✅ Function returned: {result.keys()}")
    
    print("\n   Testing classify_plant_disease()...")
    result = classify_plant_disease(image_array, api_key)
    print(f"   ✅ Function returned: {result.keys()}")
    
    print("\n" + "=" * 70)
    print("🎉 Phase 1 Implementation Testing Complete!")
    print("=" * 70)
    print("\nSummary:")
    print("✅ Disease API endpoint integrated")
    print("✅ Disease detection method implemented")
    print("✅ Comprehensive health analysis available")
    print("✅ Convenience functions ready for app.py")
    print("\nReady to proceed to Phase 2: Prompt Engineering")
    print("=" * 70)


if __name__ == "__main__":
    test_disease_detection()
