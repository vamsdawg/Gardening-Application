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
        confidence: Optional[float] = None
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
            confidence: Identification confidence (0-1)
            
        Returns:
            Dictionary with care recommendations
        """
        
        # Build the prompt
        prompt = self._build_care_prompt(
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
    
    def _build_care_prompt(
        self,
        scientific_name: str,
        common_names: list,
        family: str,
        genus: str,
        user_observation: Optional[str],
        season: Optional[str],
        confidence: Optional[float]
    ) -> str:
        """Build a comprehensive prompt for plant care recommendations"""
        
        common_names_str = ', '.join(common_names[:3]) if common_names else 'None available'
        
        prompt = f"""You are an expert horticulturist, certified arborist, and plant care specialist with deep knowledge of plant physiology, soil science, pathology, and environmental stressors. Your job is to provide thorough, actionable, and scientifically grounded plant-care recommendations.

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
            prompt += f"\nUSER'S CONCERN:\n{user_observation}\n"
            prompt += f"\nPlease address this specific concern in your recommendations.\n"
        
        prompt += """
PROVIDE CONCISE CARE RECOMMENDATIONS (keep it brief and scannable):

1. **Plant Overview** (1-2 sentences only)

2. **💧 Watering**
   - How often and how much
   - One key tip

3. **☀️ Light & Location**
   - Best placement (full sun/shade/etc.)
   - Hours of light needed

4. **🌱 Soil & Feeding**
   - Soil type
   - Fertilizer (yes/no and frequency)

5. **⚠️ Watch Out For**
   - Top 2 common problems for this plant
   - Quick fix for each
"""
        
        if user_observation:
            prompt += f"""
6. **🔧 Your Concern: "{user_observation}"**
   - Likely cause
   - What to do now (2-3 action steps max)
"""
        else:
            prompt += """
6. **💡 Key Tips**
   - 3 essential tips (one sentence each)
"""
        
        prompt += """
Keep it SHORT and practical. Use bullet points. No long paragraphs. Focus on what the user needs to DO, not botanical details.
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
    
    # Test plant care recommendations
    print("\n1. Testing Plant Care Recommendations...")
    result = llm.generate_plant_care_recommendations(
        plant_scientific_name="Mentha spicata",
        plant_common_names=["Spearmint", "Garden Mint"],
        plant_family="Lamiaceae",
        plant_genus="Mentha",
        user_observation="I noticed some brown spots on the leaf edges",
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
