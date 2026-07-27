"""Build the balanced proposer JSON for the massive_intent voice-assistant corpus."""
import json
import uuid
import datetime
import os
from pathlib import Path

SAMPLE_FILE = r"C:\Users\emily\.claude\projects\C--Users-emily-Documents-agentic-clustering\cb91043a-21cb-41ff-bf06-e5ac47f057e7\tool-results\bn6j9i3jx.txt"
WORKSPACE = Path(r"C:\Users\emily\Documents\agentic-clustering\results\clustering\massive_intent\seed=0_proposers_v2")

clusters = [
    # ===== MUSIC =====
    {
        "name": "Play specific song or artist",
        "description": "Requests to play a named song, track, or specific artist.",
        "text_ids": [
            "massive_intent-test-000195",
            "massive_intent-test-000926",
            "massive_intent-test-000142",
            "massive_intent-test-000372",
            "massive_intent-test-000989",
            "massive_intent-test-000707",
        ],
        "reasoning": "Each text names a specific artist or song; targeted music playback (not genre, playlist, or radio)."
    },
    {
        "name": "Play music by genre or mood",
        "description": "Requests to play music selected by genre, mood, or era (no specific song/artist).",
        "text_ids": [
            "massive_intent-test-000967",
            "massive_intent-test-000634",
            "massive_intent-test-001694",
            "massive_intent-test-000469",
            "massive_intent-test-000143",
            "massive_intent-test-000199",
            "massive_intent-test-001048",
            "massive_intent-test-000014",
            "massive_intent-test-000426",
        ],
        "reasoning": "Music playback where the selector is a genre/mood/era rather than a specific title."
    },
    {
        "name": "Play personal playlist",
        "description": "Requests to play a user-owned playlist or saved list of songs.",
        "text_ids": [
            "massive_intent-test-000535",
            "massive_intent-test-000603",
            "massive_intent-test-000037",
            "massive_intent-test-000435",
            "massive_intent-test-000893",
        ],
        "reasoning": "Refers explicitly to a 'playlist' (often possessive 'my'), distinct from single song or genre."
    },
    {
        "name": "Play music (generic / unspecified)",
        "description": "Generic 'play music' or 'start playing' with no specifier.",
        "text_ids": [
            "massive_intent-test-000946",
            "massive_intent-test-000590",
            "massive_intent-test-000080",
            "massive_intent-test-001610",
        ],
        "reasoning": "Music playback intent with no target specified."
    },
    {
        "name": "Play radio station",
        "description": "Requests to tune to a specific or generic radio station / FM frequency.",
        "text_ids": [
            "massive_intent-test-001608",
            "massive_intent-test-001653",
            "massive_intent-test-001632",
            "massive_intent-test-001599",
            "massive_intent-test-001612",
            "massive_intent-test-001614",
            "massive_intent-test-001640",
            "massive_intent-test-001629",
            "massive_intent-test-001604",
            "massive_intent-test-001639",
        ],
        "reasoning": "All requests target radio/FM stations specifically; distinct from on-demand music playback."
    },
    {
        "name": "Music playback control (next/previous/resume)",
        "description": "Skip, go back, resume, or move within a music track or queue.",
        "text_ids": [
            "massive_intent-test-000743",
            "massive_intent-test-000982",
            "massive_intent-test-001698",
            "massive_intent-test-002016",
        ],
        "reasoning": "Transport controls within an active playback session."
    },
    {
        "name": "Query currently playing music",
        "description": "Asking what song is currently playing or who its performer is.",
        "text_ids": [
            "massive_intent-test-000963",
            "massive_intent-test-000338",
        ],
        "reasoning": "Queries about the currently active music item."
    },
    {
        "name": "Search or discover music",
        "description": "Find / search for songs (not direct play).",
        "text_ids": [
            "massive_intent-test-000408",
            "massive_intent-test-001132",
        ],
        "reasoning": "Music discovery rather than immediate playback or named selection."
    },
    {
        "name": "Modify playlist (add/remove songs)",
        "description": "Mutate a playlist by adding, removing, or changing playback mode (e.g., shuffle).",
        "text_ids": [
            "massive_intent-test-000198",
            "massive_intent-test-001838",
        ],
        "reasoning": "Playlist-mutation intents grouped together (shuffle-mode toggling + song removal)."
    },

    # ===== PODCASTS & AUDIOBOOKS =====
    {
        "name": "Play podcast",
        "description": "Requests to play a specific podcast or podcast episode, or skip podcast episodes.",
        "text_ids": [
            "massive_intent-test-002005",
            "massive_intent-test-002003",
            "massive_intent-test-001994",
            "massive_intent-test-001969",
            "massive_intent-test-002013",
            "massive_intent-test-001649",
            "massive_intent-test-002024",
            "massive_intent-test-002008",
        ],
        "reasoning": "All concern playing or controlling podcast / episodic-audio content."
    },
    {
        "name": "Discover or list podcasts",
        "description": "Find, browse, or check for new podcasts.",
        "text_ids": [
            "massive_intent-test-001987",
            "massive_intent-test-001983",
            "massive_intent-test-001982",
        ],
        "reasoning": "Discovery / listing of podcasts rather than playback."
    },
    {
        "name": "Play audiobook",
        "description": "Requests to play a book or chapter via audio.",
        "text_ids": [
            "massive_intent-test-001679",
            "massive_intent-test-001703",
        ],
        "reasoning": "Audio-book content distinct from songs/podcasts."
    },

    # ===== ALARMS =====
    {
        "name": "Set alarm",
        "description": "Create a new alarm at a specified time/day.",
        "text_ids": [
            "massive_intent-test-000968",
            "massive_intent-test-000156",
            "massive_intent-test-000359",
            "massive_intent-test-000163",
            "massive_intent-test-000985",
            "massive_intent-test-000108",
        ],
        "reasoning": "Create-alarm intents at specific times."
    },
    {
        "name": "Remove or cancel alarm",
        "description": "Delete or turn off an existing alarm.",
        "text_ids": [
            "massive_intent-test-000582",
            "massive_intent-test-000500",
            "massive_intent-test-000243",
            "massive_intent-test-000244",
            "massive_intent-test-000284",
        ],
        "reasoning": "Alarm-removal / cancellation intents."
    },
    {
        "name": "Query alarms",
        "description": "List, review, or check what alarms are set.",
        "text_ids": [
            "massive_intent-test-000762",
            "massive_intent-test-000013",
            "massive_intent-test-000281",
            "massive_intent-test-000104",
            "massive_intent-test-000821",
            "massive_intent-test-001030",
        ],
        "reasoning": "Read-only queries about current alarm state."
    },

    # ===== REMINDERS =====
    {
        "name": "Set reminder",
        "description": "Create a reminder for a future task or event.",
        "text_ids": [
            "massive_intent-test-001280",
            "massive_intent-test-001534",
            "massive_intent-test-001275",
            "massive_intent-test-001225",
            "massive_intent-test-001320",
            "massive_intent-test-001284",
            "massive_intent-test-001531",
            "massive_intent-test-001305",
            "massive_intent-test-001207",
            "massive_intent-test-001393",
            "massive_intent-test-001587",
            "massive_intent-test-001526",
        ],
        "reasoning": "Reminder-creation intent: schedule a future nudge."
    },
    {
        "name": "Query reminders",
        "description": "Ask what reminders are set or upcoming.",
        "text_ids": [
            "massive_intent-test-001202",
            "massive_intent-test-001362",
        ],
        "reasoning": "Reminder lookup, not creation or removal."
    },

    # ===== CALENDAR / EVENTS =====
    {
        "name": "Create calendar event",
        "description": "Schedule a meeting, birthday, or appointment on the calendar.",
        "text_ids": [
            "massive_intent-test-001539",
            "massive_intent-test-001313",
            "massive_intent-test-001214",
            "massive_intent-test-001414",
            "massive_intent-test-001385",
            "massive_intent-test-001287",
            "massive_intent-test-001323",
        ],
        "reasoning": "Calendar-add intents to schedule new events."
    },
    {
        "name": "Query calendar events or meetings",
        "description": "Ask what's on the calendar, upcoming events, or past meetings.",
        "text_ids": [
            "massive_intent-test-001270",
            "massive_intent-test-001458",
            "massive_intent-test-001392",
            "massive_intent-test-001502",
            "massive_intent-test-001554",
            "massive_intent-test-002064",
            "massive_intent-test-001524",
            "massive_intent-test-002498",
        ],
        "reasoning": "Read-only event / meeting lookups."
    },
    {
        "name": "Delete or modify calendar event",
        "description": "Cancel, remove, or change an event already on the calendar.",
        "text_ids": [
            "massive_intent-test-001372",
            "massive_intent-test-001517",
            "massive_intent-test-001511",
            "massive_intent-test-001592",
            "massive_intent-test-001480",
            "massive_intent-test-001508",
        ],
        "reasoning": "Removal / edit of calendar entries."
    },

    # ===== EMAIL =====
    {
        "name": "Compose or send email",
        "description": "Write and send a new email to a recipient.",
        "text_ids": [
            "massive_intent-test-002802",
            "massive_intent-test-002936",
            "massive_intent-test-002800",
            "massive_intent-test-002758",
            "massive_intent-test-002956",
            "massive_intent-test-002763",
            "massive_intent-test-002940",
            "massive_intent-test-002934",
            "massive_intent-test-002765",
            "massive_intent-test-002923",
            "massive_intent-test-002895",
        ],
        "reasoning": "All requests authoring a new outbound email."
    },
    {
        "name": "Reply to email",
        "description": "Respond to an existing email.",
        "text_ids": [
            "massive_intent-test-002761",
            "massive_intent-test-002852",
            "massive_intent-test-002848",
            "massive_intent-test-002927",
        ],
        "reasoning": "Reply / respond intent (distinct from new compose)."
    },
    {
        "name": "Query or check email",
        "description": "Check inbox, count new emails, or look for messages from a sender.",
        "text_ids": [
            "massive_intent-test-002887",
            "massive_intent-test-002856",
            "massive_intent-test-002900",
            "massive_intent-test-002868",
            "massive_intent-test-002952",
            "massive_intent-test-002931",
            "massive_intent-test-002731",
            "massive_intent-test-002897",
            "massive_intent-test-002755",
        ],
        "reasoning": "Inbox-read intent (counting / checking / searching mail)."
    },
    {
        "name": "Manage emails (archive or delete)",
        "description": "Archive, delete, or otherwise manage existing emails.",
        "text_ids": [
            "massive_intent-test-002963",
        ],
        "reasoning": "Mailbox management distinct from compose / reply / query."
    },

    # ===== CONTACTS =====
    {
        "name": "Add to contacts",
        "description": "Add an email address or new entry to the contacts list.",
        "text_ids": [
            "massive_intent-test-002836",
            "massive_intent-test-002945",
            "massive_intent-test-002894",
        ],
        "reasoning": "Contact-creation / augmentation intent."
    },
    {
        "name": "Query contact info",
        "description": "Look up phone, email, or other contact details for a person.",
        "text_ids": [
            "massive_intent-test-002844",
            "massive_intent-test-002743",
            "massive_intent-test-002916",
        ],
        "reasoning": "Read-only contact lookups."
    },
    {
        "name": "Place phone call",
        "description": "Initiate a phone call to a contact.",
        "text_ids": [
            "massive_intent-test-002734",
        ],
        "reasoning": "Outgoing call intent."
    },

    # ===== IOT / HOME =====
    {
        "name": "Turn lights on or off",
        "description": "Switch lights on or off in a room or area.",
        "text_ids": [
            "massive_intent-test-000568",
            "massive_intent-test-000436",
            "massive_intent-test-000132",
            "massive_intent-test-000457",
            "massive_intent-test-000697",
        ],
        "reasoning": "Binary light on/off control intents."
    },
    {
        "name": "Dim or brighten lights",
        "description": "Adjust brightness of lights (dim / brighten / set level).",
        "text_ids": [
            "massive_intent-test-000176",
            "massive_intent-test-000185",
            "massive_intent-test-000970",
        ],
        "reasoning": "Continuous brightness adjustment distinct from on/off."
    },
    {
        "name": "Control smart home devices (non-light)",
        "description": "Turn on/off smart plugs, sockets, TV, or other home appliances.",
        "text_ids": [
            "massive_intent-test-000825",
            "massive_intent-test-000646",
            "massive_intent-test-002328",
        ],
        "reasoning": "IoT / appliance on-off control distinct from lights."
    },

    # ===== DEVICE / VOLUME =====
    {
        "name": "Adjust assistant volume / silence",
        "description": "Change volume of the assistant or playing audio, or tell it to stop speaking.",
        "text_ids": [
            "massive_intent-test-000803",
            "massive_intent-test-000612",
            "massive_intent-test-000184",
            "massive_intent-test-000806",
        ],
        "reasoning": "Volume control + mute / silence: same device-output-loudness intent family."
    },

    # ===== WEATHER =====
    {
        "name": "Current weather query",
        "description": "Ask about current weather conditions (here or elsewhere).",
        "text_ids": [
            "massive_intent-test-000611",
            "massive_intent-test-000714",
            "massive_intent-test-000116",
            "massive_intent-test-000875",
            "massive_intent-test-000518",
            "massive_intent-test-001004",
            "massive_intent-test-000144",
        ],
        "reasoning": "Read-only present-weather queries."
    },
    {
        "name": "Weather forecast",
        "description": "Ask about future weather (tomorrow, this week, hourly forecast).",
        "text_ids": [
            "massive_intent-test-000258",
            "massive_intent-test-000149",
            "massive_intent-test-000204",
            "massive_intent-test-000574",
        ],
        "reasoning": "Forecast / future-weather queries (distinct from current conditions)."
    },
    {
        "name": "Weather-based clothing or activity advice",
        "description": "Ask whether to bring an umbrella / jacket / sweater given the weather.",
        "text_ids": [
            "massive_intent-test-000950",
            "massive_intent-test-000377",
            "massive_intent-test-000748",
        ],
        "reasoning": "Recommendation derived from weather, distinct from raw forecast / condition queries."
    },

    # ===== NEWS =====
    {
        "name": "News headlines or updates",
        "description": "Request latest news, headlines, or coverage from a source / topic.",
        "text_ids": [
            "massive_intent-test-000616",
            "massive_intent-test-001056",
            "massive_intent-test-000114",
            "massive_intent-test-000275",
            "massive_intent-test-000238",
            "massive_intent-test-000903",
            "massive_intent-test-000150",
            "massive_intent-test-001445",
        ],
        "reasoning": "News / current-events read intents."
    },

    # ===== TIME & DATE =====
    {
        "name": "Current time query",
        "description": "Ask the current time, here or in another location.",
        "text_ids": [
            "massive_intent-test-000624",
            "massive_intent-test-000257",
            "massive_intent-test-000479",
            "massive_intent-test-000214",
            "massive_intent-test-000207",
            "massive_intent-test-000937",
        ],
        "reasoning": "Time-of-day queries."
    },
    {
        "name": "Date query",
        "description": "Ask today's date or a future date.",
        "text_ids": [
            "massive_intent-test-000422",
            "massive_intent-test-000566",
        ],
        "reasoning": "Calendar-date lookups distinct from time-of-day."
    },

    # ===== TRANSPORT / NAVIGATION =====
    {
        "name": "Navigation, directions, or traffic",
        "description": "Get directions, routing, or check current traffic conditions.",
        "text_ids": [
            "massive_intent-test-002190",
            "massive_intent-test-002215",
            "massive_intent-test-002171",
        ],
        "reasoning": "Navigation / routing / traffic queries grouped as one driving-info intent."
    },
    {
        "name": "Book taxi or ride-share",
        "description": "Order a taxi or Uber.",
        "text_ids": [
            "massive_intent-test-002152",
            "massive_intent-test-002140",
            "massive_intent-test-002154",
        ],
        "reasoning": "Ride-hailing booking intents."
    },
    {
        "name": "Book train ticket",
        "description": "Book or request a train ticket between locations.",
        "text_ids": [
            "massive_intent-test-002223",
            "massive_intent-test-002137",
            "massive_intent-test-002191",
            "massive_intent-test-002224",
            "massive_intent-test-002153",
        ],
        "reasoning": "Train-ticket booking intents."
    },
    {
        "name": "Train schedule query",
        "description": "Ask when trains arrive or depart.",
        "text_ids": [
            "massive_intent-test-002226",
            "massive_intent-test-002186",
            "massive_intent-test-002213",
            "massive_intent-test-002212",
        ],
        "reasoning": "Train-timetable lookups (no booking)."
    },
    {
        "name": "Book flight or other travel",
        "description": "Search / book flights or other long-distance travel.",
        "text_ids": [
            "massive_intent-test-002112",
            "massive_intent-test-002524",
        ],
        "reasoning": "Flight / cross-modal travel booking intents."
    },

    # ===== FOOD =====
    {
        "name": "Order food or takeout",
        "description": "Order food delivery or takeout from a restaurant.",
        "text_ids": [
            "massive_intent-test-000419",
            "massive_intent-test-001012",
            "massive_intent-test-000505",
        ],
        "reasoning": "Food-ordering / takeout intents."
    },
    {
        "name": "Find local restaurants or dining",
        "description": "Search for nearby places to eat or get a meal recommendation.",
        "text_ids": [
            "massive_intent-test-001582",
            "massive_intent-test-002088",
            "massive_intent-test-000314",
        ],
        "reasoning": "Restaurant discovery / local-dining lookups."
    },
    {
        "name": "Cooking recipe lookup",
        "description": "Ask how to cook or prepare a dish.",
        "text_ids": [
            "massive_intent-test-001759",
            "massive_intent-test-001739",
            "massive_intent-test-001755",
            "massive_intent-test-001731",
            "massive_intent-test-001774",
        ],
        "reasoning": "Recipe / cooking-how-to queries."
    },

    # ===== SOCIAL MEDIA =====
    {
        "name": "Post or tweet on social media",
        "description": "Compose and post on Twitter / Facebook including status updates.",
        "text_ids": [
            "massive_intent-test-002664",
            "massive_intent-test-002650",
            "massive_intent-test-002699",
            "massive_intent-test-002696",
            "massive_intent-test-002641",
            "massive_intent-test-002663",
        ],
        "reasoning": "Social-media outbound posting / status intents."
    },
    {
        "name": "Check social media trends or updates",
        "description": "Read trending topics or friend updates on social media.",
        "text_ids": [
            "massive_intent-test-002632",
            "massive_intent-test-002660",
            "massive_intent-test-001185",
        ],
        "reasoning": "Read-side social-media intents."
    },

    # ===== LOCAL EVENTS / MOVIES / SHOPS =====
    {
        "name": "Find local events",
        "description": "Search for events, festivals, or activities happening locally.",
        "text_ids": [
            "massive_intent-test-002079",
            "massive_intent-test-002072",
        ],
        "reasoning": "Local-events discovery (not calendar lookups, not movies)."
    },
    {
        "name": "Find or recommend movies or shows",
        "description": "Search for movies to watch or get recommendations.",
        "text_ids": [
            "massive_intent-test-002060",
            "massive_intent-test-002054",
            "massive_intent-test-002089",
            "massive_intent-test-002039",
        ],
        "reasoning": "Movie / show discovery and reviews."
    },
    {
        "name": "Find nearby shops or services",
        "description": "Search for stores or services nearby.",
        "text_ids": [
            "massive_intent-test-002044",
            "massive_intent-test-001109",
        ],
        "reasoning": "Local-business / shop discovery."
    },

    # ===== INFORMATION LOOKUP =====
    {
        "name": "Definition or dictionary lookup",
        "description": "Ask for the definition of a word.",
        "text_ids": [
            "massive_intent-test-002283",
            "massive_intent-test-002541",
        ],
        "reasoning": "Word-definition queries."
    },
    {
        "name": "Math calculation",
        "description": "Perform a numeric or arithmetic computation.",
        "text_ids": [
            "massive_intent-test-002271",
            "massive_intent-test-002302",
            "massive_intent-test-002256",
        ],
        "reasoning": "Math / arithmetic intents."
    },
    {
        "name": "Financial / market lookup (exchange rate, stocks)",
        "description": "Ask about currency exchange rates or stock / market prices.",
        "text_ids": [
            "massive_intent-test-002413",
            "massive_intent-test-002484",
            "massive_intent-test-002555",
            "massive_intent-test-002566",
        ],
        "reasoning": "Financial-data lookups: FX + stocks."
    },
    {
        "name": "Sports scores",
        "description": "Ask for sports game scores.",
        "text_ids": [
            "massive_intent-test-001138",
        ],
        "reasoning": "Sports-score intents."
    },
    {
        "name": "Facts about people or celebrities",
        "description": "Ask trivia or biographical facts about a person.",
        "text_ids": [
            "massive_intent-test-002418",
            "massive_intent-test-000237",
            "massive_intent-test-002481",
            "massive_intent-test-000362",
            "massive_intent-test-002367",
            "massive_intent-test-002348",
            "massive_intent-test-002561",
        ],
        "reasoning": "People / celebrity fact lookups."
    },
    {
        "name": "Geography or location facts",
        "description": "Ask about geographical facts (location, capital, region, statewide).",
        "text_ids": [
            "massive_intent-test-002457",
            "massive_intent-test-002400",
            "massive_intent-test-002539",
            "massive_intent-test-002588",
            "massive_intent-test-002559",
        ],
        "reasoning": "Geography facts."
    },
    {
        "name": "General factual lookup or statistics",
        "description": "Ask for an explanation, statistics, or general info about a topic (climate stats, crime stats, health tips, concept explanations).",
        "text_ids": [
            "massive_intent-test-002407",
            "massive_intent-test-002514",
            "massive_intent-test-000528",
            "massive_intent-test-000784",
            "massive_intent-test-001119",
        ],
        "reasoning": "Catch-all factual / aggregate-statistic lookups that don't fit a specialized lookup intent."
    },

    # ===== JOKES & CHITCHAT =====
    {
        "name": "Tell a joke",
        "description": "Ask the assistant for a joke.",
        "text_ids": [
            "massive_intent-test-000083",
            "massive_intent-test-000459",
            "massive_intent-test-000102",
        ],
        "reasoning": "Joke-request intents."
    },
    {
        "name": "Casual chitchat or share personal news",
        "description": "User shares something personal or makes small talk with the assistant.",
        "text_ids": [
            "massive_intent-test-001107",
            "massive_intent-test-001097",
            "massive_intent-test-001096",
            "massive_intent-test-001099",
        ],
        "reasoning": "Conversational social statements with no concrete task."
    },
    {
        "name": "Ask assistant about itself or its feelings",
        "description": "Personal questions about the assistant (feelings, opinions).",
        "text_ids": [
            "massive_intent-test-001149",
            "massive_intent-test-001154",
        ],
        "reasoning": "Self-referential questions to the assistant."
    },

    # ===== GAMES =====
    {
        "name": "Start or play game",
        "description": "Launch or play a game with the assistant.",
        "text_ids": [
            "massive_intent-test-001791",
            "massive_intent-test-001789",
            "massive_intent-test-001807",
            "massive_intent-test-001809",
            "massive_intent-test-001793",
            "massive_intent-test-001790",
            "massive_intent-test-001812",
            "massive_intent-test-001806",
        ],
        "reasoning": "Game-launching intents."
    },

    # ===== LISTS =====
    {
        "name": "Add item to list",
        "description": "Append item to a shopping, expense, or to-do list.",
        "text_ids": [
            "massive_intent-test-001862",
        ],
        "reasoning": "List-add intent."
    },
    {
        "name": "Query list contents",
        "description": "Read out what's on a to-do, work, or shopping list.",
        "text_ids": [
            "massive_intent-test-001939",
            "massive_intent-test-001826",
            "massive_intent-test-001879",
            "massive_intent-test-001902",
        ],
        "reasoning": "List-read intents."
    },
    {
        "name": "Remove item or list from lists",
        "description": "Delete an item or an entire list.",
        "text_ids": [
            "massive_intent-test-001884",
            "massive_intent-test-001821",
            "massive_intent-test-001880",
            "massive_intent-test-001941",
        ],
        "reasoning": "List-removal / list-deletion intents."
    },
]


def main():
    # Validate uniqueness
    seen = {}
    for c in clusters:
        for tid in c["text_ids"]:
            seen[tid] = seen.get(tid, 0) + 1
    dups = {k: v for k, v in seen.items() if v > 1}
    if dups:
        print("DUPLICATE assignments found:", dups)
        raise SystemExit(1)

    sample = json.load(open(SAMPLE_FILE, encoding="utf-8"))
    all_ids = [it["id"] for it in sample]
    id_to_text = {it["id"]: it["text"] for it in sample}
    assigned = set(seen.keys())
    unknown_assigned = assigned - set(all_ids)
    if unknown_assigned:
        print("ASSIGNED IDs not in sample:", unknown_assigned)
        raise SystemExit(1)
    unclustered = [tid for tid in all_ids if tid not in assigned]

    print(f"Clusters: {len(clusters)}")
    print(f"Assigned: {len(assigned)} / {len(all_ids)}")
    print(f"Unclustered: {len(unclustered)}")
    print("Unclustered texts:")
    for tid in unclustered:
        print(f"  {tid} | {id_to_text[tid]}")

    proposal = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sample_size": len(all_ids),
        "sample_strategy": "random",
        "style": "balanced",
        "existing_clusters_considered": False,
        "clusters": clusters,
        "unclustered_ids": unclustered,
        "observations": (
            "Voice-assistant utterances split cleanly along fine-grained action verbs within "
            "shared domains. Most domains break into separate CRUD-like intents (set / remove / "
            "query alarms, set / query reminders, create / query / delete calendar events, "
            "compose / reply / query / manage email, add / query contacts). Music is the most "
            "diverse domain: I split it into ten fine-grained intents (named song vs. genre vs. "
            "playlist vs. generic playback vs. radio station vs. playback control vs. discovery "
            "vs. shuffle vs. now-playing query vs. playlist mutation). Surprises: (a) many "
            "'play X' utterances target games or audiobooks rather than music, requiring a "
            "lexical disambiguation; (b) info-lookup queries fragment into distinct intents "
            "(definition, math, exchange rate, stock, sports score, person facts, geography, "
            "weather stats, crime stats, general explain) rather than one big 'Q&A' bucket; "
            "(c) weather-clothing-advice ('do i need a jacket') is functionally distinct from "
            "raw forecast / current-weather queries even though it depends on weather."
        ),
    }

    ts_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    uuid_short = uuid.uuid4().hex[:4]
    out_path = WORKSPACE / "proposals" / f"prop_{ts_str}_{uuid_short}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(proposal, f, indent=2)
    print(f"WROTE: {out_path}")


if __name__ == "__main__":
    main()
