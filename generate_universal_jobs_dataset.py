#!/usr/bin/env python3
import json, random, gzip, os
from faker import Faker

fake = Faker()
random.seed(20251024)

import json as _json
SECTORS = _json.loads("""{"Software Engineering": {"roles": ["Junior Backend Engineer", "Backend Engineer", "Senior Backend Engineer", "Frontend Engineer", "Full-Stack Engineer", "DevOps Engineer", "QA Engineer", "Engineering Manager"], "skills": ["Python", "Java", "C#", "Go", "JavaScript", "TypeScript", "React", "Node.js", "Django", "Spring Boot", "SQL", "PostgreSQL", "MongoDB", "Docker", "Kubernetes", "AWS", "Azure", "GCP", "CI/CD", "Terraform", "Linux", "Git", "REST", "GraphQL"], "certs": ["AWS Developer", "Azure Developer", "CKA", "Google Professional Cloud Developer"], "edu": ["B.Sc. Computer Science", "B.Eng. Software Engineering", "M.Sc. Computer Science"]}, "Data & Analytics": {"roles": ["Data Analyst", "BI Analyst", "Data Scientist", "Machine Learning Engineer", "Analytics Engineer", "Head of Data"], "skills": ["Python", "R", "SQL", "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "Spark", "Airflow", "dbt", "Tableau", "Power BI", "Statistics", "A/B Testing", "ETL", "Feature Engineering", "MLflow"], "certs": ["Google Data Engineer", "Databricks Lakehouse", "Tableau Desktop Specialist"], "edu": ["B.Sc. Data Science", "B.Sc. Statistics", "M.Sc. Analytics"]}, "Product Management": {"roles": ["Associate PM", "Product Manager", "Senior Product Manager", "Group PM", "Director of Product"], "skills": ["Roadmapping", "User Research", "A/B Testing", "SQL", "PRD Writing", "Agile", "Stakeholder Management", "KPIs", "Monetization", "Experimentation", "Story Mapping"], "certs": ["CSPO", "Pragmatic PMC"], "edu": ["B.Sc. Information Systems", "BBA", "MBA"]}, "Marketing": {"roles": ["Marketing Coordinator", "Content Strategist", "Growth Marketer", "SEO Manager", "Performance Marketing Manager", "Head of Marketing"], "skills": ["SEO", "SEM", "Google Ads", "Meta Ads", "Content Strategy", "Copywriting", "Email Marketing", "CRM", "HubSpot", "WordPress", "Analytics", "Funnel Optimization", "Attribution"], "certs": ["Google Ads Certification", "HubSpot Inbound", "Facebook Blueprint"], "edu": ["B.A. Marketing", "B.Com Marketing", "MBA"]}, "Sales": {"roles": ["SDR", "Account Executive", "Enterprise AE", "Sales Manager", "Head of Sales"], "skills": ["Prospecting", "Outbound", "CRM", "Salesforce", "Negotiation", "MEDDIC", "Challenger Sale", "Pipeline Management", "Forecasting", "Contracting", "RFPs", "Presentation"], "certs": ["MEDDIC", "Sandler"], "edu": ["BBA", "B.Com", "MBA"]}, "Finance & Accounting": {"roles": ["Accounting Associate", "Senior Accountant", "Financial Analyst", "FP&A Analyst", "Finance Manager", "Controller"], "skills": ["Excel", "Financial Modeling", "Valuation", "IFRS", "GAAP", "Budgeting", "Forecasting", "Variance Analysis", "Power BI", "SAP", "Oracle ERP", "Auditing"], "certs": ["CPA", "ACCA", "CFA Level I"], "edu": ["B.Com Accounting", "B.Com Finance", "MBA"]}, "Human Resources": {"roles": ["HR Coordinator", "Recruiter", "Talent Acquisition Specialist", "HR Generalist", "People Operations Manager", "HR Director"], "skills": ["Sourcing", "Boolean Search", "ATS", "Interviewing", "Onboarding", "Employer Branding", "Comp & Benefits", "Labor Law", "HRIS", "Performance Management"], "certs": ["PHR", "SHRM-CP"], "edu": ["B.A. Human Resources", "BBA", "Diploma in HR"]}, "Legal": {"roles": ["Paralegal", "Contract Specialist", "Legal Associate", "Compliance Officer", "Senior Counsel"], "skills": ["Contract Law", "Drafting", "Negotiation", "Compliance", "Regulatory", "Due Diligence", "Litigation Support", "Corporate Law", "Legal Research"], "certs": ["LLB", "LLM", "CIPP/E"], "edu": ["LLB", "LLM"]}, "Operations & Supply Chain": {"roles": ["Operations Analyst", "Supply Chain Analyst", "Operations Manager", "Logistics Coordinator", "Program Manager", "Director of Operations"], "skills": ["Lean", "Six Sigma", "Process Mapping", "ERP", "SAP", "Oracle", "Inventory", "Scheduling", "Vendor Mgmt", "Project Planning", "Risk Mgmt"], "certs": ["PMP", "Six Sigma Green Belt"], "edu": ["B.Sc. Industrial Engineering", "BBA Operations", "MBA"]}, "Design & Content": {"roles": ["Graphic Designer", "Product Designer", "UX Researcher", "Content Designer", "Design Lead"], "skills": ["Figma", "Sketch", "Adobe XD", "Illustrator", "Photoshop", "Prototyping", "User Research", "Design Systems", "Copywriting", "Accessibility"], "certs": ["HCI Certificate"], "edu": ["B.A. Graphic Design", "B.Des", "HCI Diploma"]}, "Customer Functions": {"roles": ["Customer Success Manager", "Support Specialist", "Client Services Manager", "Implementation Manager"], "skills": ["CSM", "Onboarding", "Renewals", "Churn Reduction", "NPS", "Playbooks", "Ticketing", "Zendesk", "Jira", "Communication"], "certs": ["SuccessCOACHING CSM"], "edu": ["BBA", "B.Com", "B.A. Communications"]}, "Healthcare Admin": {"roles": ["Clinical Data Analyst", "Health Informatics Analyst", "Healthcare Project Manager", "Revenue Cycle Analyst"], "skills": ["HL7", "FHIR", "EHR", "HIPAA", "SQL", "Python", "Dashboards", "Data Quality", "Clinical Workflows"], "certs": ["CPHIMS", "HIPAA Certification"], "edu": ["B.Sc. Health Informatics", "B.Sc. Public Health", "MHA"]}, "Real Estate & Facilities": {"roles": ["Property Analyst", "Facilities Coordinator", "Real Estate Financial Analyst", "FM Manager"], "skills": ["Lease Analysis", "Argus", "Excel", "Budgeting", "Vendor Mgmt", "Regulatory Compliance", "Facilities Ops"], "certs": ["RPA", "FMP"], "edu": ["B.Com Real Estate", "B.Sc. Facilities Mgmt"]}, "Education & Training": {"roles": ["Instructional Designer", "Learning Specialist", "Training Coordinator", "L&D Manager"], "skills": ["Curriculum Design", "ADDIE", "eLearning", "LMS", "Storyline", "Needs Analysis", "Assessment"], "certs": ["CPTD", "ATD Instructional Design"], "edu": ["B.Ed", "M.Ed", "Instructional Design Diploma"]}, "Consulting": {"roles": ["Business Analyst", "Associate Consultant", "Consultant", "Senior Consultant", "Engagement Manager"], "skills": ["Strategy", "Market Research", "Financial Modeling", "Process Improvement", "Client Communication", "Workshops", "Slide Design"], "certs": ["PMP", "Lean Six Sigma"], "edu": ["BBA", "B.Com", "MBA"]}}""")
SOFT_SKILLS = _json.loads("""["Teamwork", "Leadership", "Communication", "Problem Solving", "Ownership", "Adaptability", "Time Management", "Collaboration", "Mentoring", "Stakeholder Management"]""")
LANGUAGES = _json.loads("""["English", "Arabic", "French", "German", "Spanish", "Italian", "Portuguese"]""")

def rand_company():
    prefixes = ["Velora","Quantia","Bluepeak","Nexora","Orlion","Crestel","Averix","Lumena","Novanta","Polarion","Jadara","TalaTech","Meridian","Arcadia","Zafir"]
    suffixes = ["Group","Holdings","Technologies","Solutions","Partners","Industries","Labs","Systems","Networks","Advisory","Analytics"]
    return f"{random.choice(prefixes)} {random.choice(suffixes)}"

def pick_many(pool, lo, hi):
    k = random.randint(lo, hi)
    return random.sample(pool, min(k, len(pool)))

def role_level_from_title(title):
    t = title.lower()
    if any(x in t for x in ["intern","junior","associate","coordinator"]): return "Entry"
    if any(x in t for x in ["senior","lead","manager","head","director"]): return "Senior"
    return "Mid"

def synth_experience(role_title, domain_skills):
    entries = []
    num_roles = random.randint(2,4)
    start_year = random.randint(2007, 2017)
    for i in range(num_roles):
        company = rand_company()
        dur_years = random.uniform(1.0, 4.5)
        s_year = start_year + int(sum(e.get("dur",2) for e in entries))
        e_year = s_year + max(1, int(round(dur_years)))
        title_options = [role_title, "Senior "+role_title, "Associate "+role_title, role_title.replace("Senior ","")]
        title = random.choice(title_options)
        used = pick_many(domain_skills, 3, min(6, max(3, len(domain_skills)//2)))
        bullets = [
            f"Delivered {random.randint(2,6)} projects using {', '.join(used[:min(3,len(used))])} with measurable KPIs.",
            f"Improved {random.choice(['performance','reliability','conversions','accuracy','process efficiency','NPS'])} by {random.randint(5,40)}%.",
            f"Collaborated with {random.randint(3,10)} stakeholders to ship on schedule."
        ]
        entries.append({"title": title, "company": company, "start": f"{s_year}-{random.randint(1,12):02d}", "end": f"{e_year}-{random.randint(1,12):02d}", "bullets": bullets, "dur": e_year - s_year})
    for e in entries: e.pop("dur", None)
    return entries

def total_years(experience):
    tot = 0.0
    for e in experience:
        ys, ms = map(int, e["start"].split("-"))
        ye, me = map(int, e["end"].split("-"))
        tot += (ye - ys) + (me - ms)/12.0
    return max(0.5, tot)

def build_job(job_id):
    sector = random.choice(list(SECTORS.keys()))
    meta = SECTORS[sector]
    title = random.choice(meta["roles"])
    company = rand_company()
    location = random.choice(["Cairo, Egypt","Giza, Egypt","Alexandria, Egypt","Remote – EMEA","Hybrid – Cairo"])
    level = role_level_from_title(title)
    skills = meta["skills"]; certs = meta["certs"]; edu = meta["edu"]
    desc = (f"We are hiring a {title} in {sector} to drive outcomes through domain expertise and cross-functional collaboration. Success is measured by impact, quality, and delivery speed.")
    must_have = pick_many(skills, min(3, max(1, len(skills)//6)), min(6, max(3, len(skills)//2)))
    reqs = [f"Proficiency in {s}" for s in must_have]
    reqs.append(f"{random.randint(0 if level=='Entry' else 1, 2 if level=='Entry' else 7)}+ years of relevant experience")
    reqs.append("Evidence of ownership and collaboration with stakeholders")
    if random.random() < 0.6: reqs.append("Bachelor's degree or equivalent experience")
    if certs and random.random() < 0.4: reqs.append(f"Relevant certification preferred ({random.choice(certs)})")
    job = {"id": f"JOB-{job_id:06d}", "company": company, "title": title, "sector": sector, "location": location, "description": desc, "requirements": reqs}
    return job, skills, certs, edu

def score_candidate(job_reqs, cand_skills, cand_years, soft_hits):
    tech_reqs = [r for r in job_reqs if r.startswith("Proficiency in ")]
    tech_total = len(tech_reqs) if tech_reqs else 1
    tech_hits = sum(1 for r in tech_reqs if r.replace("Proficiency in ","") in cand_skills)
    tech_score = int(round((tech_hits/tech_total)*100))
    years_req = 0
    for r in job_reqs:
        if "years of relevant experience" in r:
            years_req = int(r.split("+")[0].split()[-1]); break
    exp_ratio = cand_years / (years_req if years_req>0 else 1)
    if years_req==0: exp_score = int(round(min(100, 60 + min(40, cand_years*5))))
    else: exp_score = int(round(min(96, 55 + min(35, exp_ratio*25))))
    cultural = int(round(50 + min(45, soft_hits*7)))
    overall = int(round(0.4*tech_score + 0.4*exp_score + 0.2*cultural))
    return tech_score, exp_score, cultural, overall, years_req

def detailed_breakdown(job_reqs, cand_skills, cand_years, years_req):
    tech = []
    for r in [rq for rq in job_reqs if rq.startswith("Proficiency in ")][:4]:
        skill = r.replace("Proficiency in ","")
        present = skill in cand_skills
        gap = 0 if present else random.randint(10, 30)
        tech.append({"requirement": r, "present": present, "evidence": f"Skills list includes {skill}" if present else "No mention or usage evidence in resume",
                      "gapPercentage": gap, "missingDetail": "0% gap because requirement fully met with explicit evidence." if present else f"{gap}% gap because {skill} is required; resume lists no {skill} evidence."})
    exp = []
    exp_present = cand_years >= years_req
    exp_gap = 0 if exp_present else min(40, 10*max(0, years_req - cand_years))
    exp.append({"requirement": f"{years_req}+ years of relevant experience","present":exp_present,"evidence":f"Total years ≈ {cand_years:.1f} via roles listed","gapPercentage":int(exp_gap),
                "missingDetail":"0% gap because requirement satisfied." if exp_gap==0 else f"{int(exp_gap)}% gap because role expects {years_req}+ years; resume totals ≈ {cand_years:.1f}."})
    scope_present = random.random() < 0.75
    scope_gap = 0 if scope_present else random.randint(10, 25)
    exp.append({"requirement":"Demonstrated scope and measurable impact","present":scope_present,"evidence":"Bullets quantify outcomes (%, KPIs, TCV)" if scope_present else "Bullets lack metrics",
                "gapPercentage":scope_gap,"missingDetail":"0% gap because KPIs/metrics present." if scope_gap==0 else f"{scope_gap}% gap because measurable impact required."})
    edu_arr = []
    if any("Bachelor" in r for r in job_reqs):
        edu_present = random.random() < 0.85
        edu_gap = 0 if edu_present else 20
        edu_arr.append({"requirement":"Bachelor's degree or equivalent experience","present":edu_present,"evidence":"Bachelor's degree listed" if edu_present else "No degree listed","gapPercentage":edu_gap,
                        "missingDetail":"0% gap because degree requirement satisfied." if edu_gap==0 else "20% gap because credential supports credibility; resume lacks bachelor's degree."})
    else:
        edu_arr.append({"requirement":"Formal education or equivalent experience","present":True,"evidence":"Experience considered sufficient","gapPercentage":0,"missingDetail":"0% gap."})
    soft = []
    collab_present = random.random() < 0.8
    collab_gap = 0 if collab_present else 15
    leadership_present = random.random() < 0.5
    leadership_gap = 0 if leadership_present else 10
    soft.append({"requirement":"Cross-functional collaboration","present":collab_present,"evidence":"Worked with product/ops/finance/design" if collab_present else "No collaboration examples","gapPercentage":collab_gap,
                 "missingDetail":"0% gap because collaboration evidenced." if collab_gap==0 else f"{collab_gap}% gap because cross-functional work is critical."})
    soft.append({"requirement":"Leadership/ownership behaviors","present":leadership_present,"evidence":"Owned initiatives or mentored juniors" if leadership_present else "Ownership not evidenced","gapPercentage":leadership_gap,
                 "missingDetail":"0% gap because ownership signals present." if leadership_gap==0 else "10% gap because role expects initiative ownership."})
    return tech, exp, edu_arr, soft

def build_applicant(job, domain_skills, domain_certs, domain_edu):
    name = f"{fake.first_name()} {fake.last_name()}"
    email = f"{name.lower().replace(' ','.')}{random.randint(10,9999)}@example.com"
    hard = pick_many(domain_skills, max(4, len(domain_skills)//5), max(9, len(domain_skills)//3))
    softs = pick_many(SOFT_SKILLS, 3, 6)
    extra_noise = random.sample(["Excel","Jira","Confluence","Notion","PowerPoint","Photoshop","Figma"], random.randint(0,2))
    skills = list(dict.fromkeys(hard + softs + extra_noise))
    experience = synth_experience(job["title"], domain_skills)
    years = total_years(experience)
    education = pick_many(domain_edu, 1, min(2,len(domain_edu)))
    if random.random() < 0.25 and "MBA" not in education and any("MBA" in e for e in domain_edu): education.append("MBA")
    certs = pick_many(domain_certs, 0, min(2, len(domain_certs))) if domain_certs else []
    languages = pick_many(LANGUAGES, 1, 2)
    summary = f"{job['title']} with ≈{years:.1f} years in {job['sector']} delivering measurable outcomes and collaboration."
    soft_hits = sum(1 for s in skills if s in ["Teamwork","Leadership","Communication","Ownership","Collaboration","Stakeholder Management"])
    tech_score, exp_score, cult_score, overall, years_req = score_candidate(job["requirements"], skills, years, soft_hits)
    tech_bd, exp_bd, edu_bd, soft_bd = detailed_breakdown(job["requirements"], skills, years, years_req)
    tech_reqs = [r for r in job["requirements"] if r.startswith("Proficiency in ")]
    missing_skills = [r.replace("Proficiency in ","") for r in tech_reqs if r.replace("Proficiency in ","") not in skills]
    strengths = []
    if tech_score >= 60 and tech_reqs: strengths.append("Coverage across key tools with explicit skill evidence.")
    if years >= (years_req or 0): strengths.append(f"Meets experience bar (≈{years:.1f} vs {years_req}+).")
    if cult_score >= 60: strengths.append("Collaboration/ownership evidenced via stakeholder work.")
    if not strengths: strengths.append("Adequate baseline with growth potential.")
    improvements = []
    for ms in missing_skills[:2]:
        gap = random.randint(10, 30); improvements.append(f"{ms} missing – {gap}%% gap because it is a core requirement; resume has no {ms}.")
    if years < (years_req or 0):
        diff = (years_req or 0) - years; g = min(40, int(diff*10)); improvements.append(f"Years short by ≈{diff:.1f} – {g}%% gap vs {years_req}+; resume totals ≈{years:.1f}.")
    if not improvements: improvements.append("Increase quantification of impact to strengthen seniority signal.")
    match_summary = (f"Alignment on {len(tech_reqs)-len(missing_skills)}/{len(tech_reqs)} core tools; experience ≈{years:.1f} vs {years_req}+ requirement. "
                     f"Strengths include collaboration and ownership; gaps focus on {', '.join(missing_skills[:2]) if missing_skills else 'minor scope signals'}.")
    analysis = {"overallScore": overall, "technicalSkillsScore": tech_score, "experienceScore": exp_score, "culturalFitScore": cult_score,
                 "matchSummary": match_summary, "strengthsHighlights": strengths[:3], "improvementAreas": improvements[:3],
                 "detailedBreakdown": {"technicalSkills": tech_bd, "experience": exp_bd, "educationAndCertifications": edu_bd, "culturalFitAndSoftSkills": soft_bd}}
    if random.random() < 0.07 and years < (years_req or 0):
        analysis["redFlags"] = [{"issue":"Seniority mismatch","evidence":f"Years ≈ {years:.1f} (< {years_req}+ req)","reason":"Limited autonomy risk."}]
    return {"candidate": {"name": name, "email": email, "summary": summary, "skills": skills, "experience": experience, "education": education,
            "certifications": certs, "languages": languages, "location": random.choice(["Cairo, Egypt","Giza, Egypt","Alexandria, Egypt","Remote – EMEA"])}, "analysis": analysis, "grade": overall}

def make_job_record(job_id, applicants_per_job=10):
    job, skills, certs, edu = build_job(job_id)
    applicants = [build_applicant(job, skills, certs, edu) for _ in range(applicants_per_job)]
    return {"job": job, "applicants": applicants}

def main():
    os.makedirs(".", exist_ok=True)
    shards = 40; jobs_per_shard = 500; applicants_per_job = 10
    job_id = 1
    for shard_idx in range(1, shards+1):
        path = f"universal_jobs_shard_{shard_idx:02d}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as gz:
            for _ in range(jobs_per_shard):
                rec = make_job_record(job_id, applicants_per_job)
                gz.write(json.dumps(rec, ensure_ascii=False) + "\n")
                job_id += 1
        print("Wrote", path)

if __name__ == "__main__":
    main()
