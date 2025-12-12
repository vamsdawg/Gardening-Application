"""
PlantNet API Integration for Plant Identification
"""
import requests
import json
from io import BytesIO
from PIL import Image
import numpy as np

class PlantNetAPI:
    def __init__(self, api_key):
        """
        Initialize PlantNet API client
        
        Args:
            api_key: Your PlantNet API key from https://my.plantnet.org/
        """
        self.api_key = api_key
        self.base_url = "https://my-api.plantnet.org/v2/identify/all"
        self.disease_url = "https://my-api.plantnet.org/v2/diseases/identify"
        
    def identify_from_array(self, image_array, num_results=3):
        """
        Identify a plant from a numpy array
        
        Args:
            image_array: numpy array of the image (RGB)
            num_results: number of top results to return (default: 3)
            
        Returns:
            dict with identification results or None if error
        """
        try:
            # Convert numpy array to PIL Image
            if isinstance(image_array, np.ndarray):
                image = Image.fromarray(image_array.astype('uint8'), 'RGB')
            else:
                image = image_array
            
            # Convert to bytes
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            # Prepare the request
            files = [('images', ('image.jpg', img_byte_arr, 'image/jpeg'))]
            params = {
                'api-key': self.api_key,
                'include-related-images': 'false'
            }
            
            # Make the API request
            response = requests.post(
                self.base_url,
                files=files,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_response(data, num_results)
            elif response.status_code == 404:
                return {
                    'success': False,
                    'error': 'No plant identified',
                    'message': 'PlantNet could not identify this plant. Try a clearer image.'
                }
            elif response.status_code == 401:
                return {
                    'success': False,
                    'error': 'Invalid API key',
                    'message': 'Please check your PlantNet API key.'
                }
            else:
                return {
                    'success': False,
                    'error': f'API error: {response.status_code}',
                    'message': response.text
                }
                
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'Request timeout',
                'message': 'PlantNet API took too long to respond. Please try again.'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'Error during identification: {str(e)}'
            }
    
    def _parse_response(self, data, num_results):
        """Parse PlantNet API response into a clean format"""
        if 'results' not in data or len(data['results']) == 0:
            return {
                'success': False,
                'error': 'No results',
                'message': 'No matching plants found.'
            }
        
        results = []
        for i, result in enumerate(data['results'][:num_results]):
            species = result.get('species', {})
            
            plant_info = {
                'rank': i + 1,
                'scientific_name': species.get('scientificNameWithoutAuthor', 'Unknown'),
                'scientific_name_full': species.get('scientificName', 'Unknown'),
                'common_names': species.get('commonNames', []),
                'family': species.get('family', {}).get('scientificNameWithoutAuthor', 'Unknown'),
                'genus': species.get('genus', {}).get('scientificNameWithoutAuthor', 'Unknown'),
                'confidence': result.get('score', 0.0),
                'confidence_pct': f"{result.get('score', 0.0) * 100:.1f}%"
            }
            
            results.append(plant_info)
        
        return {
            'success': True,
            'top_result': results[0],
            'all_results': results,
            'query_info': {
                'remaining_credits': data.get('remainingIdentificationRequests', 'Unknown'),
                'best_match': results[0]['scientific_name']
            }
        }
    
    def format_plant_name(self, result):
        """
        Format plant name for display
        
        Args:
            result: A single plant result dict
            
        Returns:
            Formatted string with common name and scientific name
        """
        scientific = result.get('scientific_name', 'Unknown')
        common_names = result.get('common_names', [])
        
        if common_names:
            # Get English common name if available, otherwise first one
            common_name = common_names[0] if common_names else None
            return f"{common_name} ({scientific})"
        else:
            return scientific
    
    def identify_disease_from_array(self, image_array, organs='auto', num_results=5, include_images=False):
        """
        Identify plant diseases from a numpy array
        
        Args:
            image_array: numpy array of the image (RGB)
            organs: plant organ type (DEPRECATED - disease API auto-detects organ)
            num_results: number of top results to return (default: 5)
            include_images: whether to include related disease images (default: False)
            
        Returns:
            dict with disease identification results or None if error
        """
        try:
            # Convert numpy array to PIL Image
            if isinstance(image_array, np.ndarray):
                image = Image.fromarray(image_array.astype('uint8'), 'RGB')
            else:
                image = image_array
            
            # Convert to bytes
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            # Prepare the request - organs should NOT be in params for disease API
            files = [('images', ('image.jpg', img_byte_arr, 'image/jpeg'))]
            
            # Only include API key and optional parameters in query string
            params = {
                'api-key': self.api_key,
                'include-related-images': 'true' if include_images else 'false',
                'nb-results': num_results
            }
            
            # Note: organs parameter is not supported by disease API
            # It auto-detects the organ type
            
            # Make the API request
            response = requests.post(
                self.disease_url,
                files=files,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_disease_response(data, num_results)
            elif response.status_code == 404:
                return {
                    'success': True,
                    'has_disease': False,
                    'message': 'No diseases detected - plant appears healthy',
                    'all_results': [],
                    'remaining_credits': None
                }
            elif response.status_code == 401:
                return {
                    'success': False,
                    'error': 'Invalid API key',
                    'message': 'Please check your PlantNet API key.'
                }
            else:
                return {
                    'success': False,
                    'error': f'API error: {response.status_code}',
                    'message': response.text
                }
                
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'Request timeout',
                'message': 'PlantNet Disease API took too long to respond. Please try again.'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'Error during disease identification: {str(e)}'
            }
    
    def _parse_disease_response(self, data, num_results):
        """Parse PlantNet Disease API response into a clean format"""
        if 'results' not in data or len(data['results']) == 0:
            return {
                'success': True,
                'has_disease': False,
                'message': 'No diseases detected - plant appears healthy',
                'all_results': [],
                'remaining_credits': data.get('remainingIdentificationRequests', 'Unknown')
            }
        
        diseases = []
        for i, result in enumerate(data['results'][:num_results]):
            disease_info = {
                'rank': i + 1,
                'name': result.get('name', 'Unknown'),  # EPPO code
                'label': result.get('description', result.get('label', 'Unknown')),  # Disease description
                'confidence': result.get('score', 0.0),
                'confidence_pct': f"{result.get('score', 0.0) * 100:.1f}%",
                'images': result.get('images', [])[:3] if result.get('images') else []  # Limit to 3 reference images
            }
            
            diseases.append(disease_info)
        
        # Determine if there's a significant disease detection
        top_disease = diseases[0]
        has_significant_disease = top_disease['confidence'] > 0.3  # 30% threshold
        
        return {
            'success': True,
            'has_disease': has_significant_disease,
            'top_disease': top_disease,
            'all_results': diseases,
            'query_info': {
                'remaining_credits': data.get('remainingIdentificationRequests', 'Unknown'),
                'version': data.get('version', 'Unknown'),
                'organs_analyzed': data.get('query', {}).get('organs', ['auto'])
            },
            'message': f"Detected: {top_disease['label']}" if has_significant_disease else "No significant diseases detected"
        }
    
    def analyze_plant_health(self, image_array, num_species_results=3, num_disease_results=5):
        """
        Comprehensive plant analysis: identifies both species AND diseases
        
        Args:
            image_array: numpy array of the image (RGB)
            num_species_results: number of species matches to return
            num_disease_results: number of disease matches to return
            
        Returns:
            dict with both species and disease information
        """
        # Identify species
        species_result = self.identify_from_array(image_array, num_species_results)
        
        # Identify diseases
        disease_result = self.identify_disease_from_array(image_array, num_results=num_disease_results)
        
        # Combine results
        return {
            'species': species_result,
            'disease': disease_result,
            'combined_success': species_result.get('success', False) or disease_result.get('success', False)
        }


# Convenience function for integration with existing app.py code
def classify_plant_with_health(image_array, api_key):
    """
    Convenience function for comprehensive plant analysis (species + disease)
    Compatible with existing app.py workflow
    
    Args:
        image_array: numpy array of the image (RGB)
        api_key: PlantNet API key
        
    Returns:
        dict with comprehensive analysis results
    """
    api = PlantNetAPI(api_key)
    return api.analyze_plant_health(image_array)


def classify_plant_disease(image_array, api_key, organs='auto', num_results=5):
    """
    Convenience function for disease identification only
    Compatible with existing app.py workflow
    
    Args:
        image_array: numpy array of the image (RGB)
        api_key: PlantNet API key
        organs: plant organ type ('leaf', 'flower', 'fruit', 'bark', or 'auto')
        num_results: number of results to return
        
    Returns:
        dict with disease identification results
    """
    api = PlantNetAPI(api_key)
    return api.identify_disease_from_array(image_array, organs, num_results)


def test_api(api_key, test_image_path):
    """
    Test the PlantNet API with an image
    
    Args:
        api_key: Your PlantNet API key
        test_image_path: Path to test image
    """
    from PIL import Image
    import numpy as np
    
    print("Testing PlantNet API...")
    print(f"API Key: {api_key[:10]}...")
    
    # Load test image
    image = Image.open(test_image_path).convert('RGB')
    image_array = np.array(image)
    
    # Initialize API
    api = PlantNetAPI(api_key)
    
    print("\n" + "="*60)
    print("TEST 1: Species Identification")
    print("="*60)
    
    # Identify plant
    print("Sending species identification request...")
    result = api.identify_from_array(image_array, num_results=3)
    
    # Display results
    if result['success']:
        print("\n✅ Species Identification Success!")
        print(f"\nTop Result: {result['top_result']['scientific_name']}")
        print(f"Common names: {', '.join(result['top_result']['common_names']) or 'None'}")
        print(f"Confidence: {result['top_result']['confidence_pct']}")
        print(f"Family: {result['top_result']['family']}")
        
        print(f"\nAll matches:")
        for r in result['all_results']:
            common = ', '.join(r['common_names']) if r['common_names'] else 'No common name'
            print(f"  {r['rank']}. {r['scientific_name']} ({common}) - {r['confidence_pct']}")
        
        print(f"\nRemaining API credits: {result['query_info']['remaining_credits']}")
    else:
        print(f"\n❌ Error: {result['error']}")
        print(f"Message: {result['message']}")
    
    print("\n" + "="*60)
    print("TEST 2: Disease Identification")
    print("="*60)
    
    # Identify diseases
    print("Sending disease identification request...")
    disease_result = api.identify_disease_from_array(image_array, num_results=5)
    
    if disease_result['success']:
        if disease_result['has_disease']:
            print("\n⚠️ Disease Detection Success!")
            print(f"\nTop Result: {disease_result['top_disease']['label']}")
            print(f"EPPO Code: {disease_result['top_disease']['name']}")
            print(f"Confidence: {disease_result['top_disease']['confidence_pct']}")
            
            print(f"\nAll disease matches:")
            for d in disease_result['all_results']:
                print(f"  {d['rank']}. {d['label']} (Code: {d['name']}) - {d['confidence_pct']}")
        else:
            print("\n✅ No diseases detected - plant appears healthy!")
        
        print(f"\nRemaining API credits: {disease_result['query_info']['remaining_credits']}")
        print(f"API Version: {disease_result['query_info']['version']}")
    else:
        print(f"\n❌ Error: {disease_result['error']}")
        print(f"Message: {disease_result['message']}")
    
    print("\n" + "="*60)
    print("TEST 3: Comprehensive Health Analysis")
    print("="*60)
    
    # Comprehensive analysis
    print("Running comprehensive plant health analysis...")
    health_result = api.analyze_plant_health(image_array)
    
    if health_result['combined_success']:
        print("\n✅ Comprehensive Analysis Complete!")
        
        # Species info
        if health_result['species']['success']:
            species = health_result['species']['top_result']
            print(f"\n🌿 SPECIES: {species['scientific_name']}")
            if species['common_names']:
                print(f"   Common: {species['common_names'][0]}")
            print(f"   Confidence: {species['confidence_pct']}")
        
        # Disease info
        if health_result['disease']['success']:
            if health_result['disease']['has_disease']:
                disease = health_result['disease']['top_disease']
                print(f"\n⚠️ HEALTH STATUS: Issue Detected")
                print(f"   Disease: {disease['label']}")
                print(f"   Confidence: {disease['confidence_pct']}")
            else:
                print(f"\n✅ HEALTH STATUS: Appears Healthy")
    else:
        print("\n❌ Analysis failed")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Test with your API key
    API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual key
    
    # Test with an image (update path)
    test_image = "path/to/test/image.jpg"
    
    # Uncomment to test:
    # test_api(API_KEY, test_image)
    print("PlantNet API module loaded. Set your API key to use.")
