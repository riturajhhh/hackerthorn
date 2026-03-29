from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_insight(row):
    prompt = f"""
    Explain SLA risk and suggest action:
    Priority: {row['priority']}
    Time Remaining: {row['time_remaining']}
    Probability: {row['probability']}
    """

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}]
        )
        return res.choices[0].message.content
    except:
        return "Error"