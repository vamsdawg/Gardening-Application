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
    
    # Identify plant
    print("Sending identification request...")
    result = api.identify_from_array(image_array, num_results=3)
    
    # Display results
    if result['success']:
        print("\n✅ Success!")
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


if __name__ == "__main__":
    # Test with your API key
    API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual key
    
    # Test with an image (update path)
    test_image = "path/to/test/image.jpg"
    
    # Uncomment to test:
    # test_api(API_KEY, test_image)
    print("PlantNet API module loaded. Set your API key to use.")
