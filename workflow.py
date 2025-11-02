"""
FAISS-based Job Matching System with LLM Evaluation
Vectorizes jobs, stores them in FAISS, finds best matches, and evaluates with LLM.
"""


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from sentence_transformers import SentenceTransformer
import numpy as np
import json
import random
import faiss
import pickle
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# ============================================================================
# LOAD CONFIGURATION
# ============================================================================

load_dotenv()


import json

# Input and output file paths
input_file = 'universal_jobs_shard_01.jsonl'
output_file = 'jobs_only.json'

# List to store all jobs
jobs = []

# Read the JSONL file line by line
with open(input_file, 'r', encoding='utf-8') as infile:
    for line in infile:
        # Parse each line as JSON
        data = json.loads(line)
        
        # Extract only the job information (without applicants)
        if 'job' in data:
            jobs.append(data['job'])

# Write the jobs to a new JSON file
with open(output_file, 'w', encoding='utf-8') as outfile:
    json.dump(jobs, outfile, indent=2, ensure_ascii=False)

print(f"Extracted {len(jobs)} jobs to {output_file}")


import json

def get_all_applicant_from_json(file_path='universal_jobs_shard_01.json'):
    all_applicants = []
    with open(file_path, 'r') as file:
        data = json.load(file)
        for job in data:
            all_applicants += job.get('applicants', [])
    return all_applicants


# Get all applicants
all_applicants = get_all_applicant_from_json(
    "universal_jobs_shard_01.json"
)

# Print the count
print(f"Total applicants: {len(all_applicants)}")

# Dump results into a JSON file
output_path = "applications.json"
with open(output_path, "w") as outfile:
    json.dump(all_applicants, outfile, indent=4, ensure_ascii=False)

print(f"All applicants have been saved to '{output_path}' ✅")




SYSTEM_PROMPT = """


You are a precise job-candidate matching system. Your task is to evaluate candidates against job requirements using strict, objective criteria.

## EVALUATION FRAMEWORK

### 1. SKILLS ASSESSMENT
For each required skill in the job posting:
- **MATCHED**: Candidate explicitly lists this skill in their Skills section OR demonstrates it through project work in Experience section
- **NOT MATCHED**: Skill is neither listed nor demonstrated in any capacity
- **PARTIAL**: Skill is demonstrated indirectly (e.g., "customer management" for "CSM") - treat as MATCHED only if the connection is clear and direct

### 2. EXPERIENCE ASSESSMENT
Calculate total relevant experience by:
- Sum all roles that match the job's sector/function (e.g., "Customer Functions" matches "Customer Success Manager" roles)
- Use the date ranges provided (YYYY-MM format)
- Round to one decimal place (e.g., 6.1 years)
- Compare against requirement using: ≥ for "X+ years", exact match for "X years", range check for "X-Y years"

### 3. QUALIFICATION ASSESSMENT
- **Education**: Check if candidate's degree level meets or exceeds requirement (PhD > Master's > Bachelor's > Associate)
- **Certifications**: List any relevant certifications if mentioned
- **Location**: MATCHED if candidate location matches or is within job location region

### 4. EVIDENCE OF REQUIREMENTS
For qualitative requirements (e.g., "ownership", "collaboration", "stakeholder management"):
- **MATCHED**: Explicit evidence in experience bullets (e.g., "Collaborated with X stakeholders")
- **NOT MATCHED**: No mention in any experience section
- Use exact keywords or clear synonyms only

## OUTPUT FORMAT

Return your evaluation in this exact JSON structure:

{
  "skills_evaluation": {
    "required_skills": [
      {
        "skill": "<exact skill name from job posting>",
        "status": "MATCHED|NOT_MATCHED",
        "evidence": "<where found: 'Skills section' or 'Experience: [Company]' or 'Not found'>"
      }
    ],
    "skills_match_rate": "<X/Y matched>"
  },
  "experience_evaluation": {
    "required_years": <number>,
    "candidate_years": <number>,
    "status": "MATCHED|NOT_MATCHED",
    "calculation": "<brief explanation of how years were calculated>"
  },
  "qualitative_requirements": [
    {
      "requirement": "<exact requirement from job posting>",
      "status": "MATCHED|NOT_MATCHED",
      "evidence": "<specific quote from CV or 'Not found'>"
    }
  ],
  "location_match": {
    "required": "<job location>",
    "candidate": "<candidate location>",
    "status": "MATCHED|NOT_MATCHED"
  },
  "overall_assessment": {
    "total_requirements": <number>,
    "requirements_met": <number>,
    "match_percentage": <number>,
    "recommendation": "STRONG_MATCH|MODERATE_MATCH|WEAK_MATCH|NO_MATCH"
  }
}

## MATCHING RULES

1. **Case Insensitive**: Treat "Jira" and "jira" as identical
2. **Exact Match Priority**: Prefer exact keyword matches over interpretations
3. **No Assumptions**: Do not infer skills or experience not explicitly stated
4. **Date Calculations**: 
   - 1 year = 12 months
   - Calculate months between dates, then convert to years with 1 decimal
   - Example: 2007-10 to 2008-12 = 14 months = 1.2 years
5. **Synonym Handling**: Only accept clear synonyms:
   - "CSM" = "Customer Success Manager" = "Customer Success Management"
   - "Ticketing" = "Ticketing Systems" = "Ticket Management"
   - Do NOT expand beyond obvious synonyms
6. **Recommendation Thresholds**:
   - STRONG_MATCH: ≥85% requirements met
   - MODERATE_MATCH: 60-84% requirements met
   - WEAK_MATCH: 40-59% requirements met
   - NO_MATCH: <40% requirements met

## CONSISTENCY GUIDELINES

- Always process requirements in the order they appear in the job posting
- Use the same evidence extraction logic for all candidates
- When multiple interpretations are possible, choose the most conservative (stricter) one
- Do not add commentary outside the JSON structure
- Ensure all numbers are calculated identically across evaluations

## CRITICAL: DETERMINISM ENFORCERS

- Do not use probabilistic language ("likely", "possibly", "seems to")
- Do not make subjective judgments about quality or fit
- Base ALL decisions on explicit text matching and arithmetic operations
- If a requirement is ambiguous, note it in evidence but follow the strictest interpretation

</system_prompt>

"""

# ============================================================================
# LLM INITIALIZATION
# ============================================================================

def init_llm(provider):
    """Initialize LLM based on provider."""
    if provider == 'openai':
        return ChatOpenAI(
            model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
            temperature=float(os.getenv('TEMPERATURE', 0.0)),
            seed=int(os.getenv('SEED', 42)),
            api_key=os.getenv('OPENAI_API_KEY'),
            top_p=0,
        )
    elif provider == 'gemini':
        return ChatGoogleGenerativeAI(
            model=os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp'),
            temperature=float(os.getenv('TEMPERATURE', 0.0)),
            max_retries=2,
            api_key=os.getenv('GOOGLE_API_KEY')
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_jobs_from_json(filename):
    """Load jobs data from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def load_applications_from_json(filename):
    """Load applications data from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

# ============================================================================
# TEXT CONVERSION
# ============================================================================

def job_to_text(job):
    """Convert job JSON to searchable text"""
    parts = [
        job.get('title', ''),
        job.get('sector', ''),
        job.get('location', ''),
        job.get('description', ''),
        ' '.join(job.get('requirements', []))
    ]
    return ' '.join(parts)

def application_to_text(candidate):
    """Convert candidate JSON to searchable text"""
    experience_text = " ".join([
        f"{exp.get('title', '')} at {exp.get('company', '')} {' '.join(exp.get('bullets', []))}"
        for exp in candidate.get('experience', [])
    ])
    
    skills_text = ' '.join(candidate.get('skills', []))
    education_text = ' '.join(candidate.get('education', []))
    summary_text = candidate.get('summary', '')
    location_text = candidate.get('location', '')
    
    return f"{summary_text} {skills_text} {experience_text} {education_text} {location_text}"

# ============================================================================
# LLM EVALUATION FUNCTIONS
# ============================================================================

def format_experience(experience_list):
    """Format candidate experience for the prompt."""
    formatted = []
    for exp in experience_list:
        formatted.append(
            f"• {exp.get('title')} at {exp.get('company')} "
            f"({exp.get('start')} to {exp.get('end')})"
        )
        for bullet in exp.get('bullets', []):
            formatted.append(f"  - {bullet}")
    return '\n'.join(formatted)

def create_evaluation_prompt(candidate, job):
    """Create the evaluation prompt for the LLM."""
    prompt = f"""# JOB POSTING
Title: {job.get('title')}
Company: {job.get('company')}
Location: {job.get('location')}
Sector: {job.get('sector')}

Description: {job.get('description')}

Requirements:
{chr(10).join('- ' + r for r in job.get('requirements', []))}

---

# CANDIDATE
Name: {candidate.get('name')}
Location: {candidate.get('location')}
Summary: {candidate.get('summary')}

Skills: {', '.join(candidate.get('skills', []))}

Experience:
{format_experience(candidate.get('experience', []))}

Education: {', '.join(candidate.get('education', []))}

---

Evaluate this match and respond with JSON only."""
    
    return prompt

def parse_llm_response(response_text):
    """Extract and parse JSON from LLM response."""
    if '```json' in response_text:
        response_text = response_text.split('```json')[1].split('```')[0]
    elif '```' in response_text:
        response_text = response_text.split('```')[1].split('```')[0]
    
    return json.loads(response_text.strip())

import time
import uuid



def match_candidate_to_job(candidate, job, llm, provider_name):
    """Evaluate how well a candidate matches a job posting."""
    prompt = create_evaluation_prompt(candidate, job)
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    
    try:
        result = parse_llm_response(response.content)
        os.makedirs(f'logs', exist_ok=True)
        with open(f"logs/{uuid.uuid4()}.json", 'w') as f: json.dump(result, f, indent=2)
        
        return result
        
    except Exception as e:
        print(f"error in LLM call : {e}")
        exit()
        return {
            'error': str(e),
            'raw_response': response.content,
            'candidate_name': candidate.get('name'),
            'job_title': job.get('title')
        }

# ============================================================================
# FAISS INDEX CREATION & MANAGEMENT
# ============================================================================

def create_faiss_index(embeddings, jobs_data):
    """Create FAISS index from embeddings with job metadata."""
    if embeddings.shape[0] != len(jobs_data):
        raise ValueError(
            f"Mismatch: {embeddings.shape[0]} embeddings but {len(jobs_data)} jobs. "
            "They must be in the same order!"
        )
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    jobs_metadata = []
    for i, job in enumerate(jobs_data):
        jobs_metadata.append({
            'faiss_index': i,
            'job_id': job.get('id'),
            'job_data': job
        })
    
    print(f"✓ FAISS index created with {len(jobs_metadata)} jobs")
    print("  Index Validation (first 3 jobs):")
    for i in range(min(3, len(jobs_metadata))):
        print(f"    FAISS index {i} -> Job ID: {jobs_metadata[i]['job_id']} - {jobs_metadata[i]['job_data'].get('title')}")
    
    return index, jobs_metadata

def find_most_similar_jobs(application_embedding, faiss_index, jobs_metadata, top_k=5):
    """Find most similar jobs using FAISS."""
    application_embedding = application_embedding.reshape(1, -1).astype('float32')
    faiss.normalize_L2(application_embedding)
    
    similarities, indices = faiss_index.search(application_embedding, top_k)
    
    results = []
    for rank, (similarity, faiss_idx) in enumerate(zip(similarities[0], indices[0]), 1):
        job_meta = jobs_metadata[faiss_idx]
        
        results.append({
            'rank': rank,
            'similarity_score': float(similarity),
            'faiss_index': int(faiss_idx),
            'job_id': job_meta['job_id'],
            'job': job_meta['job_data']
        })
    
    return results

def save_faiss_system(faiss_index, jobs_metadata, index_path="jobs_faiss_index.bin", 
                      metadata_path="jobs_metadata.pkl"):
    """Save FAISS index and metadata for later use"""
    faiss.write_index(faiss_index, index_path)
    with open(metadata_path, 'wb') as f:
        pickle.dump(jobs_metadata, f)
    print(f"✓ Saved FAISS index to {index_path}")
    print(f"✓ Saved metadata to {metadata_path}")

def load_faiss_system(index_path="jobs_faiss_index.bin", metadata_path="jobs_metadata.pkl"):
    """Load FAISS index and metadata"""
    faiss_index = faiss.read_index(index_path)
    with open(metadata_path, 'rb') as f:
        jobs_metadata = pickle.load(f)
    print(f"✓ Loaded FAISS index from {index_path}")
    print(f"✓ Loaded metadata for {len(jobs_metadata)} jobs")
    return faiss_index, jobs_metadata

# ============================================================================
# CONSISTENCY METRICS
# ============================================================================
def calculate_consistency_metrics(results):
    """
    Calculate consistency metrics from multiple evaluation runs.
    
    Args:
        results: List of evaluation result dictionaries
        
    Returns:
        Dictionary with consistency metrics
    """
    # Filter out error results
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        print("❌ No valid results to analyze")
        return None
    
    # Extract scores from the new JSON structure
    scores = [r['overall_assessment']['match_percentage'] for r in valid_results]
    
    if not scores:
        print("❌ No scores found in results")
        return None
    
    # Calculate basic statistics
    from collections import Counter
    import statistics
    
    score_counts = Counter(scores)
    mode_score = score_counts.most_common(1)[0][0]
    mode_frequency = score_counts.most_common(1)[0][1]
    
    metrics = {
        'total_runs': len(results),
        'valid_runs': len(valid_results),
        'error_runs': len(results) - len(valid_results),
        'scores': scores,
        'mode_score': mode_score,
        'mode_frequency': mode_frequency,
        'min_score': min(scores),
        'max_score': max(scores),
        'score_range': max(scores) - min(scores),
        'avg_score': statistics.mean(scores),
        'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0.0,
        'median_score': statistics.median(scores),
    }
    
    # Add recommendation consistency analysis
    recommendations = [r['overall_assessment']['recommendation'] for r in valid_results]
    rec_counts = Counter(recommendations)
    
    metrics['recommendations'] = {
        'all': recommendations,
        'distribution': dict(rec_counts),
        'mode_recommendation': rec_counts.most_common(1)[0][0],
        'mode_recommendation_frequency': rec_counts.most_common(1)[0][1],
        'consistency_rate': rec_counts.most_common(1)[0][1] / len(recommendations) * 100
    }
    
    # Add skills evaluation consistency
    skills_matched_counts = []
    for r in valid_results:
        skills_eval = r['skills_evaluation']['skills_match_rate']
        # Parse "X/Y matched" format
        matched, total = skills_eval.split('/')[0], skills_eval.split('/')[1].split()[0]
        skills_matched_counts.append(int(matched))
    
    if skills_matched_counts:
        metrics['skills_consistency'] = {
            'matched_counts': skills_matched_counts,
            'mode_matched': Counter(skills_matched_counts).most_common(1)[0][0],
            'variance': statistics.stdev(skills_matched_counts) if len(skills_matched_counts) > 1 else 0.0
        }
    
    # Add requirements met consistency
    requirements_met = [r['overall_assessment']['requirements_met'] for r in valid_results]
    total_requirements = [r['overall_assessment']['total_requirements'] for r in valid_results]
    
    metrics['requirements_consistency'] = {
        'requirements_met': requirements_met,
        'total_requirements': total_requirements[0] if total_requirements else 0,
        'mode_requirements_met': Counter(requirements_met).most_common(1)[0][0],
        'variance': statistics.stdev(requirements_met) if len(requirements_met) > 1 else 0.0
    }
    
    # Overall consistency score (0-100, where 100 is perfect consistency)
    consistency_score = 100 - (metrics['std_dev'] * 10)  # Penalize std dev
    consistency_score -= (metrics['score_range'] / 2)  # Penalize range
    consistency_score = max(0, min(100, consistency_score))  # Clamp to 0-100
    
    metrics['consistency_score'] = round(consistency_score, 2)
    
    return metrics

def display_detailed_metrics(metrics):
    """Pretty print detailed metrics"""
    print("\n" + "="*80)
    print("DETAILED CONSISTENCY ANALYSIS")
    print("="*80)
    
    print(f"\n📊 SCORE STATISTICS:")
    print(f"   Runs:              {metrics['valid_runs']}/{metrics['total_runs']} successful")
    print(f"   Score Range:       {metrics['min_score']}% - {metrics['max_score']}%")
    print(f"   Average:           {metrics['avg_score']:.2f}%")
    print(f"   Median:            {metrics['median_score']}%")
    print(f"   Mode:              {metrics['mode_score']}% ({metrics['mode_frequency']} times)")
    print(f"   Std Deviation:     {metrics['std_dev']:.2f}")
    print(f"   Consistency Score: {metrics['consistency_score']}/100")
    
    print(f"\n🎯 RECOMMENDATION CONSISTENCY:")
    for rec, count in metrics['recommendations']['distribution'].items():
        pct = count / metrics['valid_runs'] * 100
        print(f"   {rec}: {count}/{metrics['valid_runs']} ({pct:.1f}%)")
    print(f"   Primary: {metrics['recommendations']['mode_recommendation']} "
          f"({metrics['recommendations']['consistency_rate']:.1f}% agreement)")
    
    print(f"\n🔧 SKILLS MATCHING CONSISTENCY:")
    print(f"   Mode Skills Matched: {metrics['skills_consistency']['mode_matched']}")
    print(f"   Variance:            {metrics['skills_consistency']['variance']:.2f}")
    
    print(f"\n✓ REQUIREMENTS CONSISTENCY:")
    print(f"   Total Requirements:     {metrics['requirements_consistency']['total_requirements']}")
    print(f"   Mode Requirements Met:  {metrics['requirements_consistency']['mode_requirements_met']}")
    print(f"   Variance:               {metrics['requirements_consistency']['variance']:.2f}")
    
    print("="*80)

def calculate_consistency_metrics_(results):
    """Calculate consistency metrics from multiple evaluation results."""
    scores = [r.get('overall_score', 0) for r in results if 'error' not in r]
    
    if not scores:
        return None
    
    min_score = min(scores)
    max_score = max(scores)
    avg_score = sum(scores) / len(scores)
    
    variance = sum((x - avg_score)**2 for x in scores) / len(scores)
    std_dev = variance**0.5
    
    differences = []
    for i in range(len(scores)):
        for j in range(i+1, len(scores)):
            differences.append(abs(scores[i] - scores[j]))
    
    max_difference = max(differences) if differences else 0
    avg_difference = sum(differences) / len(differences) if differences else 0
    
    # Calculate mode (most frequent score)
    from collections import Counter
    score_counts = Counter(scores)
    mode_score = score_counts.most_common(1)[0][0]
    mode_frequency = score_counts.most_common(1)[0][1]
    
    return {
        'scores': scores,
        'min_score': min_score,
        'max_score': max_score,
        'avg_score': avg_score,
        'mode_score': mode_score,
        'mode_frequency': mode_frequency,
        'std_dev': std_dev,
        'score_range': max_score - min_score,
        'max_difference': max_difference,
        'avg_difference': avg_difference,
        'num_runs': len(scores)
    }


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

import json
import os

def print_matching_results(candidate, similar_jobs, output_file="results.json"):
    """Print matching results in a nice format and save to JSON file"""

    # ---------- Print Section ----------
    print("\n" + "="*80)
    print("CANDIDATE PROFILE")
    print("="*80)
    print(f"Name: {candidate.get('name', 'N/A')}")
    print(f"Location: {candidate.get('location', 'N/A')}")
    print(f"Summary: {candidate.get('summary', 'N/A')}")
    skills = candidate.get('skills', [])
    if len(skills) > 5:
        print(f"Skills: {', '.join(skills[:5])}...")
    else:
        print(f"Skills: {', '.join(skills)}")
    
    print("\n" + "="*80)
    print("TOP MATCHING JOBS")
    print("="*80)
    
    for result in similar_jobs:
        job = result['job']
        print(f"\n🏆 Rank {result['rank']}: Similarity Score: {result['similarity_score']:.4f}")
        print(f"   Job ID: {result['job_id']}")
        print(f"   Title: {job.get('title', 'N/A')}")
        print(f"   Company: {job.get('company', 'N/A')}")
        print(f"   Location: {job.get('location', 'N/A')}")
        print(f"   Sector: {job.get('sector', 'N/A')}")
        if job.get('requirements'):
            print(f"   Requirements: {job['requirements'][0]}..." if len(job['requirements']) > 0 else "")
        print("-" * 80)
    
    # ---------- Save Section ----------
    # Prepare structured data for saving
    data_to_save = {
        "candidate": candidate,
        "matching_jobs": similar_jobs
    }

    # Load existing results if file exists
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    # Append new candidate results
    existing_data.append(data_to_save)

    # Save updated data back to file
    with open(output_file, "w") as f:
        json.dump(existing_data, f, indent=4)

    print(f"\n✅ Results saved to '{output_file}'")


def print_matching_results_(candidate, similar_jobs):
    """Print matching results in a nice format"""
    print("\n" + "="*80)
    print("CANDIDATE PROFILE")
    print("="*80)
    print(f"Name: {candidate.get('name', 'N/A')}")
    print(f"Location: {candidate.get('location', 'N/A')}")
    print(f"Summary: {candidate.get('summary', 'N/A')}")
    skills = candidate.get('skills', [])
    if len(skills) > 5:
        print(f"Skills: {', '.join(skills[:5])}...")
    else:
        print(f"Skills: {', '.join(skills)}")
    
    print("\n" + "="*80)
    print("TOP MATCHING JOBS")
    print("="*80)
    
    for result in similar_jobs:
        job = result['job']
        print(f"\n🏆 Rank {result['rank']}: Similarity Score: {result['similarity_score']:.4f}")
        print(f"   Job ID: {result['job_id']}")
        print(f"   Title: {job.get('title', 'N/A')}")
        print(f"   Company: {job.get('company', 'N/A')}")
        print(f"   Location: {job.get('location', 'N/A')}")
        print(f"   Sector: {job.get('sector', 'N/A')}")
        if job.get('requirements'):
            print(f"   Requirements: {job['requirements'][0]}..." if len(job['requirements']) > 0 else "")
        print("-" * 80)



# ============================================================================
# MAIN SCRIPT
# ============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("FAISS JOB MATCHING SYSTEM WITH LLM EVALUATION")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Load and vectorize jobs
    # ========================================================================
    
    print("\n📂 Loading jobs data...")
    jobs_data = load_jobs_from_json("jobs_only.json")
    print(f"✓ Loaded {len(jobs_data)} jobs")
    
    print("\n🧠 Loading embedding model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("✓ Model loaded successfully!")
    
    print("\n🔄 Vectorizing all jobs...")
    job_embeddings = []
    
    for i, job in enumerate(jobs_data):
        job_text = job_to_text(job)
        job_embedding = model.encode(job_text)
        job_embeddings.append(job_embedding)
        
        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{len(jobs_data)} jobs")
    
    job_embeddings = np.array(job_embeddings).astype('float32')
    print(f"✓ Created embeddings matrix: {job_embeddings.shape}")
    
    # ========================================================================
    # STEP 2: Create FAISS index
    # ========================================================================
    
    print("\n🗄️  Creating FAISS index with job metadata...")
    faiss_index, jobs_metadata = create_faiss_index(job_embeddings, jobs_data)
    
    print("\n💾 Saving FAISS system...")
    save_faiss_system(faiss_index, jobs_metadata)
    
    # ========================================================================
    # STEP 3: Load test candidate
    # ========================================================================
    
    try:
        print("\n📂 Loading applications data...")
        applications_data = load_applications_from_json("applications.json")
        print(f"✓ Loaded {len(applications_data)} applications")
        
        random_application_obj = random.choice(applications_data)
        random_candidate = random_application_obj['candidate']
        
        print(f"\n🎯 Selected random candidate: {random_candidate.get('name', 'Unknown')}")
        
    except FileNotFoundError:
        print("\n⚠️  Applications file not found. Using sample candidate...")
        
        random_candidate = {
            "name": "Sample Candidate",
            "skills": ["Python", "Machine Learning", "Data Analysis", "TensorFlow"],
            "experience": [
                {
                    "title": "Data Scientist",
                    "company": "Tech Corp",
                    "start": "2020-01",
                    "end": "2023-12",
                    "bullets": ["Built ML models", "Analyzed data"]
                }
            ],
            "education": ["Computer Science degree"],
            "summary": "Experienced developer with focus on AI and data science",
            "location": "San Francisco"
        }
    
    # ========================================================================
    # STEP 4: Find matching jobs with FAISS
    # ========================================================================
    
    print("\n🔍 Vectorizing candidate profile...")
    candidate_text = application_to_text(random_candidate)
    candidate_embedding = model.encode(candidate_text)
    
    print("🔍 Searching for best matching jobs...")
    similar_jobs = find_most_similar_jobs(
        candidate_embedding, 
        faiss_index, 
        jobs_metadata, 
        top_k=5
    )
    
    print_matching_results(random_candidate, similar_jobs)
    
    # ========================================================================
    # STEP 5: LLM Evaluation of TOP match (5 times)
    # ========================================================================
    
    print("\n" + "="*80)
    print("LLM EVALUATION - TOP MATCH (10 RUNS)")
    print("="*80)
    
    top_job = similar_jobs[0]['job']
    top_similarity = similar_jobs[0]['similarity_score']
    
    print(f"\n🏆 TOP MATCHED JOB:")
    print(f"   Title: {top_job.get('title')}")
    print(f"   Company: {top_job.get('company')}")
    print(f"   FAISS Similarity: {top_similarity:.4f}")
    
    print("\n🤖 Initializing OpenAI LLM...")
    openai_llm = init_llm('openai')
    
    print("\n🔄 Running 10 evaluations...")
    results = []
    
    for i in range(10):
        print(f"   Run {i+1}/10...", end=" ")
        result = match_candidate_to_job(random_candidate, top_job, openai_llm, 'openai')
        result['run_number'] = i + 1
        results.append(result)
        
        if 'error' not in result:
            print(f"Score: {result['overall_assessment']['match_percentage']}")
        else:
            print(f"ERROR")
    
    # Display results
    print("\n" + "-"*80)
    print("INDIVIDUAL RESULTS:")
    print("-"*80)
    
    for result in results:
        if 'error' not in result:
            print(f"Run {result['run_number']}: Score = {result['overall_assessment']['match_percentage']}, "
                  f"Recommendation = {result.get('recommendation')}")
        else:
            print(f"Run {result['run_number']}: ERROR - {result.get('error')}")
    
    # Calculate metrics
    metrics = calculate_consistency_metrics(results)
    if metrics:
        display_detailed_metrics(metrics)
