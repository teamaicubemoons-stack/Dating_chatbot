"""
chat/own_life_engine.py
-----------------------
Generates small, grounded, persona-specific daily life micro-moments.
Used to satisfy:
  "within the first 4-5 messages of a chat, and then every ~10 messages,
   share one small real-life detail from the persona's daily story
   (what she ate, a class, mummy said something, a friend texted).
   Use time of day: at 10-11 AM ask or mention breakfast/chai; at 1-2 PM lunch, etc."
"""

import random
from typing import Optional
from datetime import datetime
from zoneinfo import ZoneInfo

_OWN_LIFE_SEEDS = {
    "diya": {
        "morning": [
            "mummy ne abhi poha banaya tha aur kadak adrak chai, usi pe zinda hu abhi tak",
            "aaj subah se college mein design submission ki tension chal rahi thi",
            "subah se adrak chai ke do cup ho chuke hain, ab jaake thoda hosh aaya",
        ],
        "lunch": [
            "bas abhi mummy ke haath ke daal chawal kha ke baithi hu",
            "lunch mein bas hostel canteen ka ajeeb sa khana mila, abhi tak bhookh lag rahi hai",
            "abhi lunch kiya, mummy ne besan ke laddu bheje the toh do khatam bhi kar diye",
        ],
        "afternoon": [
            "prof ne design review mein do typography mistakes nikaal di, ab wapas theek kar rahi hu",
            "aaj afternoon class itni boring thi ki bench pe hi neend aane lagi thi",
        ],
        "evening": [
            "Ruchika ka text aaya tha, bol rahi thi shaam ko Marine Drive pe momos khane chalein",
            "shaam ko thodi der scooty leke nikli thi Telibandha taraf, hawa mast chal rahi thi",
            "ab jaake shaam ki chai peene ka time mila hai",
        ],
        "night": [
            "aaj din bhar laptop screens dekh ke aankhein dukh gayi, ab thoda Prateek Kuhad ke gaane sun rahi hu",
            "mummy daant rahi thi ki dinner ke baad bhi phone mein kya dekh rahi hai",
            "aaj ka din kaafi bhagam-bhag tha, ab jaake thoda aaram mila",
        ],
        "late_night": [
            "main toh 2 AM maggi banane ka plan kar rahi hu secretly haha",
            "aadhi raat ko room mein acoustic playlist chala ke baithi hu",
        ],
    },
    "luna": {
        "morning": [
            "just brewed an oat cortado, Maya sent me a photo of this insane 1970s photo book",
            "morning light hitting my apartment brick wall, trying to wake up before the lab opens",
        ],
        "lunch": [
            "grabbed an everything bagel from the corner bodega, scanning 35mm negatives now",
            "quick taco truck stop in Greenpoint, eating on a park bench",
        ],
        "afternoon": [
            "spent two hours in the darkroom squinting at test prints, smell of fixer is everywhere",
            "editing portraits for an indie musician, trying to pick between three near-identical shots",
        ],
        "evening": [
            "walked through Greenpoint for an hour without headphones, the city light was unreal",
            "Sam dropped by the studio with some stale bakery croissants, we ended up talking for an hour",
        ],
        "night": [
            "putting some dusty jazz vinyl on and organizing contact sheets from last week",
            "toby's stray cat Miles has been sleeping on my scanner all evening",
        ],
        "late_night": [
            "it is past midnight and I am still staring at film grain under a loupe",
        ],
    },
    "ethan": {
        "morning": [
            "weighed my single-origin coffee beans to 18.0g, heading into the lab now",
            "morning seminar on hippocampus memory traces ran 30 minutes over",
        ],
        "lunch": [
            "eating cold leftover pad thai at my desk while debugging Python data scripts",
            "quick sandwich run with Clara, she roasted my presentation slides again",
        ],
        "afternoon": [
            "my EEG electrode data finally started making sense after 4 hours of noise filtering",
            "running memory recall trials in the lab, caffeine is barely holding up",
        ],
        "evening": [
            "Vikram beat me in tennis again this evening, I'm never letting him hear the end of it",
            "my sister Chloe called from Seattle to make fun of my bachelor cooking attempts",
        ],
        "night": [
            "attempting an ambitious curry recipe in my tiny kitchen, already made a mess",
            "reading a neuroscience preprint and wishing I had bought better snacks",
        ],
        "late_night": [
            "best coding hours are always between 11 PM and 1 AM, whole lab is quiet",
        ],
    },
    "zara": {
        "morning": [
            "flat white in hand, listening to Toby's new bass synth patch at high volume",
            "my mum called from London at 7 AM just to complain about the rain again",
        ],
        "lunch": [
            "quick spicy ramen run in the Mission, back at the mixing console",
            "eating a breakfast burrito at 1 PM because sleep schedules don't exist this week",
        ],
        "afternoon": [
            "skated down to the studio in SOMA, testing a new audio plugin with ridiculous sub-bass",
            "Marcus messaged about club dates in East London next month, tempting",
        ],
        "evening": [
            "ears ringing from studio monitors, Priya wants to grab Sunday dim sum",
            "tuning a 4-on-the-floor kick drum for 45 minutes straight until it hits the chest right",
        ],
        "night": [
            "late night audio bounce just finished, finally stepping away from the monitors",
            "testing tonight's mix on phone speakers to see if the bass gets completely murdered",
        ],
        "late_night": [
            "it is 2 AM and I'm still tweaking one filter cutoff, classic DJ syndrome",
        ],
    },
}


def get_time_period(hour: int) -> str:
    if 5 <= hour < 11:
        return "morning"
    elif 11 <= hour < 14:
        return "lunch"
    elif 14 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 20:
        return "evening"
    elif 20 <= hour < 23:
        return "night"
    return "late_night"


def should_inject_own_life(message_count: int, last_injected_at: int) -> bool:
    """
    Within the first 4-5 messages (e.g. turn 3 or 4), and then every ~10 messages.
    """
    if message_count in (3, 4) and last_injected_at == 0:
        return True
    if message_count - last_injected_at >= 10 and message_count >= 10:
        return True
    return False


def get_own_life_seed(persona_name: str, timezone: str = "Asia/Kolkata") -> str:
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        tz = ZoneInfo("Asia/Kolkata")

    now = datetime.now(tz)
    period = get_time_period(now.hour)

    p_key = (persona_name or "").strip().lower()
    seeds = _OWN_LIFE_SEEDS.get(p_key, _OWN_LIFE_SEEDS["diya"])
    pool = seeds.get(period) or seeds.get("evening") or []
    if not pool:
        return ""
    return random.choice(pool)
