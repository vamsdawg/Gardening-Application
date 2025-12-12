"""
Google Gemini LLM Integration for Plant Care Recommendations
"""
import google.generativeai as genai
import json
from typing import Dict, Optional


class PlantCareLLM:
    def __init__(self, api_key: str):
        """
        Initialize Google Gemini LLM client
        
        Args:
            api_key: Your Google Gemini API key from https://aistudio.google.com/app/apikey
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')  # Free, fast model
        
    def generate_plant_care_recommendations(
        self,
        plant_scientific_name: str,
        plant_common_names: list,
        plant_family: str,
        plant_genus: str,
        user_observation: Optional[str] = None,
        season: Optional[str] = None,
        confidence: Optional[float] = None,
        disease_name: Optional[str] = None,
        disease_confidence: Optional[float] = None
    ) -> Dict:
        """
        Generate comprehensive plant care recommendations
        
        Args:
            plant_scientific_name: Scientific name from PlantNet
            plant_common_names: List of common names
            plant_family: Plant family
            plant_genus: Plant genus
            user_observation: User's description of issues/concerns
            season: Current season
            confidence: Species identification confidence
            disease_name: Disease name if detected
            disease_confidence: Disease detection confidence
            
        Returns:
            Dictionary with care recommendations
        """
        
        # Build the appropriate prompt based on disease detection
        if disease_name and disease_confidence and disease_confidence > 0.3:
            # Disease detected - use treatment-focused prompt
            prompt = self._build_diseased_plant_prompt(
                plant_scientific_name,
                plant_common_names,
                plant_family,
                plant_genus,
                disease_name,
                disease_confidence,
                user_observation,
                season,
                confidence
            )
        else:
            # No disease - use general care prompt
            prompt = self._build_healthy_plant_prompt(
                plant_scientific_name,
                plant_common_names,
                plant_family,
                plant_genus,
                user_observation,
                season,
                confidence
            )
        
        try:
            # Generate response
            response = self.model.generate_content(prompt)
            
            # Parse and structure the response
            recommendations = self._parse_response(response.text)
            
            return {
                'success': True,
                'recommendations': recommendations,
                'raw_response': response.text
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'Error generating recommendations: {str(e)}'
            }
    
    def _build_diseased_plant_prompt(
        self,
        scientific_name: str,
        common_names: list,
        family: str,
        genus: str,
        disease_name: str,
        disease_confidence: float,
        user_observation: Optional[str],
        season: Optional[str],
        species_confidence: Optional[float]
    ) -> str:
        """Build treatment-focused prompt for diseased plants"""
        
        common_names_str = ', '.join(common_names[:3]) if common_names else 'None available'
        
        prompt = f"""You are a plant pathologist and disease specialist. A plant disease has been DETECTED in the uploaded image.

PLANT IDENTIFICATION:
- Scientific Name: {scientific_name}
- Common Names: {common_names_str}
- Family: {family}
- Genus: {genus}
"""
        
        if species_confidence:
            prompt += f"- Species Confidence: {species_confidence*100:.1f}%\n"
        
        prompt += f"""
⚠️ DISEASE DETECTED:
- Disease/Issue: {disease_name}
- Detection Confidence: {disease_confidence*100:.1f}%
"""
        
        if season:
            prompt += f"- Current Season: {season.capitalize()}\n"
        
        if user_observation:
            prompt += f"\nUSER'S OBSERVATIONS:\n{user_observation}\n"
        
        prompt += f"""

IMPORTANT: This plant has a CONFIRMED health issue. Be EXTREMELY CONCISE.

Provide BRIEF treatment advice (EXACTLY 3 bullets per section, short sentences):

1. **🔍 Issue**
   - What: {disease_name} in one sentence
   - Severity: minor/moderate/serious for {scientific_name}
   - Spreads: yes/no

2. **🚨 Do Now** (next 24-48 hours)
   - [Critical action 1]
   - [Critical action 2]
   - [Critical action 3]

3. **💊 Treatment**
   - Product 1: [name, $price]
   - Product 2: [name, $price]
   - How: [brief application method]

4. **📅 Recovery**
   - Timeline: [X weeks]
   - Good sign 1: [what to look for]
   - Good sign 2: [what to look for]

5. **🛡️ Prevent**
   - [Top prevention tip]
   - [Environmental change needed]
   - [Ongoing maintenance]

STRICT RULES: 
- Each bullet = ONE short sentence (max 10 words)
- EXACTLY 3 bullets per section
- NO explanations or theory
- Action words only
"""
        
        return prompt
    
    def _build_healthy_plant_prompt(
        self,
        scientific_name: str,
        common_names: list,
        family: str,
        genus: str,
        user_observation: Optional[str],
        season: Optional[str],
        confidence: Optional[float]
    ) -> str:
        """Build care-focused prompt for healthy plants"""
        
        common_names_str = ', '.join(common_names[:3]) if common_names else 'None available'
        
        prompt = f"""You are an expert horticulturist. The user has uploaded an image of their plant seeking care advice.

PLANT IDENTIFICATION:
- Scientific Name: {scientific_name}
- Common Names: {common_names_str}
- Family: {family}
- Genus: {genus}
"""
        
        if confidence:
            prompt += f"- Identification Confidence: {confidence*100:.1f}%\n"
        
        if season:
            prompt += f"\nCURRENT CONTEXT:\n- Season: {season.capitalize()}\n"
        
        if user_observation:
            prompt += f"\nUSER'S QUESTION/CONCERN:\n{user_observation}\n"
        
        prompt += f"""

✅ NO DISEASES DETECTED. Be EXTREMELY CONCISE.

Provide care advice (EXACTLY 3 bullets per section, short sentences):

1. **💧 Water**
   - Frequency: [how often in {season if season else 'now'}]
   - Amount: [how much]
   - Tip: [one key mistake to avoid]

2. **☀️ Light**
   - Location: [ideal spot]
   - Hours: [number needed]
   - Warning: [sign of too much/little]

3. **🌱 Soil & Feed**
   - Soil: [type needed]
   - Fertilizer: [schedule for {season if season else 'now'}]
   - Ratio: [NPK if specific, or frequency]

4. **⚠️ Watch For** ({season if season else 'year-round'})
   - Problem 1: [name + quick fix]
   - Problem 2: [name + quick fix]
   - Problem 3: [name + quick fix]
"""
        
        if user_observation:
            prompt += f"""
5. **🔧 Your Question**
   "{user_observation}"
   - [Brief answer in 3 bullets max]
"""
        else:
            prompt += f"""
5. **💡 Quick Tips**
   - [Essential tip 1]
   - [Essential tip 2]
   - [Essential tip 3]
"""
        
        prompt += """

STRICT RULES:
- Each bullet = ONE short sentence (max 10 words)
- EXACTLY 3 bullets per section
- NO long explanations
- Direct and actionable only
"""
        
        return prompt
    
    def _parse_response(self, response_text: str) -> str:
        """
        Parse and clean the LLM response
        Could be enhanced to extract structured data
        """
        # For now, return the full text
        # In future, could parse into structured sections
        return response_text
    
    def generate_diagnosis(
        self,
        plant_scientific_name: str,
        plant_common_names: list,
        symptoms: str,
        season: Optional[str] = None
    ) -> Dict:
        """
        Generate a plant problem diagnosis
        
        Args:
            plant_scientific_name: Scientific name
            plant_common_names: Common names
            symptoms: Description of the problem
            season: Current season
            
        Returns:
            Dictionary with diagnosis and treatment
        """
        
        common_names_str = ', '.join(plant_common_names[:3]) if plant_common_names else 'Unknown'
        
        prompt = f"""You are a plant pathologist diagnosing plant problems.

PLANT: {plant_scientific_name} ({common_names_str})
SEASON: {season.capitalize() if season else 'Unknown'}

SYMPTOMS REPORTED:
{symptoms}

Provide a structured diagnosis:

1. **Most Likely Cause** (state the single most probable cause)
   - Confidence level (High/Medium/Low)
   - Why you think this is the cause

2. **Alternative Possibilities** (list 2-3 other potential causes)

3. **Diagnostic Steps**
   - How to confirm the diagnosis
   - What to look for

4. **Treatment Plan**
   - Immediate actions (today/this week)
   - Short-term treatment (1-2 weeks)
   - Long-term prevention

5. **Expected Timeline**
   - How long until improvement
   - Signs of recovery

6. **When to Worry**
   - Red flags that indicate the problem is serious
   - When to consider removing the plant

Be specific to THIS plant species and these symptoms. Prioritize practical, home-gardener solutions.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return {
                'success': True,
                'diagnosis': response.text
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


def test_gemini_integration(api_key: str):
    """Test the Gemini LLM integration"""
    
    print("Testing Google Gemini Integration...")
    print("=" * 60)
    
    # Initialize LLM
    llm = PlantCareLLM(api_key)
    
    # Test 1: Healthy plant care
    print("\n1. Testing Healthy Plant Care Recommendations...")
    result = llm.generate_plant_care_recommendations(
        plant_scientific_name="Mentha spicata",
        plant_common_names=["Spearmint", "Garden Mint"],
        plant_family="Lamiaceae",
        plant_genus="Mentha",
        user_observation="I want to keep my plant healthy",
        season="summer",
        confidence=0.95
    )
    
    if result['success']:
        print("✅ SUCCESS!")
        print("\nRecommendations:")
        print(result['recommendations'])
    else:
        print(f"❌ ERROR: {result['error']}")
    
    print("\n" + "=" * 60)
    
    # Test 2: Diseased plant treatment
    print("\n2. Testing Diseased Plant Treatment Recommendations...")
    result = llm.generate_plant_care_recommendations(
        plant_scientific_name="Solanum lycopersicum",
        plant_common_names=["Tomato", "Garden Tomato"],
        plant_family="Solanaceae",
        plant_genus="Solanum",
        disease_name="Powdery Mildew",
        disease_confidence=0.85,
        season="summer",
        confidence=0.92
    )
    
    if result['success']:
        print("✅ SUCCESS!")
        print("\nTreatment Plan:")
        print(result['recommendations'])
    else:
        print(f"❌ ERROR: {result['error']}")
    
    print("\n" + "=" * 60)
    
    # Test diagnosis
    print("\n2. Testing Problem Diagnosis...")
    diagnosis = llm.generate_diagnosis(
        plant_scientific_name="Mentha spicata",
        plant_common_names=["Spearmint"],
        symptoms="Leaves are turning yellow and have small brown spots",
        season="summer"
    )
    
    if diagnosis['success']:
        print("✅ SUCCESS!")
        print("\nDiagnosis:")
        print(diagnosis['diagnosis'])
    else:
        print(f"❌ ERROR: {diagnosis['error']}")


if __name__ == "__main__":
    # Test with your API key
    API_KEY = "YOUR_GEMINI_API_KEY_HERE"
    
    # Uncomment to test:
    # test_gemini_integration(API_KEY)
    print("Gemini LLM module loaded. Set your API key to test.")
