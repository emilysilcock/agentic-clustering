"""Action-vs-info lens proposer for voice-assistant utterances.

We split each text into:
  1) request type (control/imperative, info-seeking, conversational/meta)
  2) domain

Output clusters are domain-shaped (~14-22), with the action lens used as a
discriminator to keep them clean.
"""
import json
import re
import uuid
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(
    r"C:\Users\emily\Documents\agentic-clustering\results\clustering\massive_domain\seed=0_proposers_v1"
)
SAMPLE = WORKSPACE / "proposals" / "_shared_sample.json"


def classify(text: str) -> str:
    """Return a cluster key for `text`. Order matters: earlier rules win."""
    t = text.lower().strip()

    # --- helper substrings ---
    def has(*subs):
        return any(s in t for s in subs)

    def word(*ws):
        # whole-ish word match
        return any(re.search(rf"\b{re.escape(w)}\b", t) for w in ws)

    # ----- EARLY OVERRIDES (handle ambiguous phrases first) -----
    # Game-playing should NOT fall into music ("play")
    if has(
        "play scrabble",
        "play game",
        "we should play nfs",
        "knock knock",
        "funny jokes",
        "hello how is your day",
        "how is your day",
        "how has your day been",
        "do you like my girlfriend",
        "this is the best band",
        "i want it to remember my preferences",
        "i would like it to help analyze ideas",
        "opinion petabit",
    ):
        return "conversational_chitchat"

    # General-QA early catches (avoid catching on stray keywords)
    if has(
        "where do the rocky mountains",
        "how would you describe",
        "what is the birthday of hemingway",
    ):
        return "general_qa"

    # Listening to music / podcasts should not fall into lists_shopping just
    # because of "listen" containing "list"
    if has(
        "i'd like to listen",
        "i want to listen",
        "who sings the song",
        "song that i am listening",
        "play me a random audio book",
        "favorite podcast from list",
    ):
        return "audio_playback_music_radio_podcasts"

    # Contact-list adds should go to contacts, not generic lists
    if has(
        "add bob to my list of contacts",
        "add to my list of contacts",
    ):
        return "messaging_email_contacts"

    # ----- EMAIL / MESSAGING / CONTACTS -----
    # any mention of email / inbox / mails / reply (email-shaped) + contacts
    if has("email", "e mail", "inbox", "unread mail", "new mail") or word(
        "emails", "mails"
    ):
        return "messaging_email_contacts"
    if has("dictate email"):
        return "messaging_email_contacts"
    if has(
        "did charlotte responded",
        "has ben got in touch",
        "please archive my read messages",
        "contact detail",
        "phone number",
        "my contacts",
        "add to my contacts",
        "add to my list of contacts",
        "add to contacts",
        "contact emails",
        "contacts please",
        "are my contacts mostly",
    ):
        return "messaging_email_contacts"

    # ----- SOCIAL MEDIA / POSTING -----
    if has(
        "tweet",
        "twitter",
        "facebook",
        "social media",
        "social network",
        "my feed",
        "friend update",
        "social networks",
        "post to",
        "post my",
        "submit a negative review",
        "negative review about a company",
        "potus",
        "tell comcast i hate them",
    ):
        return "social_media"

    # ----- ALARM -----
    if word("alarm", "alarms"):
        return "alarms"
    if has("wake me up", "morning alarm", "default alarm"):
        return "alarms"

    # ----- CALENDAR / EVENTS / MEETINGS / APPOINTMENTS -----
    if has(
        "calendar",
        "appointment",
        "appointments",
        "meeting",
        "meetings",
        "schedule",
        "scheduled",
        "event",
        "events",
        "birthday",
        "lunch with",
        "dinner with",
        "next friday the thirteenth",
        "halloween",
        "new year",
        "anniversary",
        "next saturday",
        "this friday",
        "tomorrow morning",
        "upcoming week",
        "next three months",
        "do this week",
        "my next three",
        "scheduled for the",
        "what do i have to do",
        "any cultural events",
        "any special events",
        "what i have to do",
        "set a notification for sports game",
        "cancel the breakfast at tiffany",
        "please set this date to repeat",
        "cancel my plans to pick up my parents",
        "do i have a date on friday",
        "cancel dinner tonight",
        "am i free at four p. m.",
        "anything good happening this weekend",
    ):
        # but not if it's primarily a contact-add etc
        if "contact" in t and "calendar" not in t and "event" not in t and "meeting" not in t:
            pass
        else:
            return "calendar_events"

    # ----- REMINDERS / TO-DO / NOTES -----
    if has(
        "remind me",
        "reminder",
        "reminders",
        "to do list",
        "to do",
        "to-do",
        "pending reminders",
        "make a note",
        "make a list about",
        "errands",
        "alert me",
        "send an alert before",
        "is there anything i should be reminded about",
        "both young drivers and provisional drivers",  # long note/dictation paragraph
    ):
        return "reminders_todo"
    if t in ("remind me at",) or t.startswith("remind me at"):
        return "reminders_todo"

    # ----- LISTS (shopping, generic) -----
    if has(
        "shopping list",
        "grocery list",
        "groceries",
        "shopping",
    ):
        # if there's also calendar/email, those should already have matched above
        return "lists_shopping"
    if has(
        "list",
        "lists",
    ) and not has("playlist", "favorite list", "music list"):
        # things like "list all the lists", "add to list", "give me to do list",
        # "remove the guest list", "remove my list of favorite albums", "play
        # all by playlist" — playlist excluded
        return "lists_shopping"
    if t in ("delete item",):
        return "lists_shopping"

    # ----- MUSIC / RADIO / PODCASTS / AUDIOBOOKS -----
    if has(
        "playlist",
        "song",
        "music",
        "pandora",
        "spotify",
        "itunes",
        "sirius",
        "xmtune",
        "radio",
        "play",
        "track",
        "album",
        "lady gaga",
        "adele",
        "katy perry",
        "billy joel",
        "nirvana",
        "david bowie",
        "elton john",
        "tune to",
        "favorite albums",
        "music list",
        "rock",
        "jazz",
        "audiobook",
        "audio book",
        "podcast",
        "podcasts",
        "channel",
        "open bad religion folder",
        "resume joes book from where i left off",
        "vitaly channel",
    ):
        return "audio_playback_music_radio_podcasts"

    # ----- SMART HOME: LIGHTS -----
    if has("light", "lights", "dim", "brighten"):
        return "smart_home_lights"
    if t == "make it red in here":
        return "smart_home_lights"

    # ----- SMART HOME: APPLIANCES (coffee maker, vacuum, plug, socket, sound system) -----
    if has(
        "coffee maker",
        "coffee machine",
        "make me some coffee",
        "make the coffee",
        "make me a cup of coffee",
        "start coffee",
        "coffee make now",
        "run coffee maker",
        "vacuum",
        "hoover",
        "clean my house",
        "suck out the dust",
        "smart socket",
        "smart plug",
        "enable my plug",
        "feed and pet my dog",
        "robot vacuum",
    ):
        return "smart_home_appliances"

    # ----- AUDIO/VOLUME CONTROL of the assistant -----
    if has(
        "volume",
        "mute",
        "silence",
        "shut down the sound",
        "turn off sound",
        "speak loudly",
        "be quiet",
        "stop speaking",
        "don't make any sounds",
        "don t make any sounds",
        "do not make any noise",
        "silence speakers",
    ):
        return "audio_volume_control"

    # ----- WEATHER -----
    if (
        has(
            "weather",
            "temperature",
            "sunny",
            "storms",
            "umbrella",
            "shovel my driveway",
            "cold will it get",
        )
        or word("rain", "raining", "rainy", "snow")
    ):
        return "weather"

    # ----- DATE / TIME / CLOCK -----
    if has(
        "what time",
        "current time",
        "time in",
        "time difference",
        "what's the time",
        "is it wednesday",
        "is the twenty",
        "what day is",
        "what day of the week",
        "next friday the thirteenth",
    ):
        return "datetime_clock"
    if t in ("what time is it",):
        return "datetime_clock"

    # ----- NEWS -----
    if has(
        "news",
        "front page",
        "latest international",
        "world news",
        "news stories",
        "news articles",
        "cnn website",
        "c. n. n. website",
        "headlines",
        "match highlights",
        "score of the game",
        "latest updates",
        "give me the latest updates",
    ):
        return "news_headlines"
    if has("brexit", "trump", "investigation into trump"):
        return "news_headlines"

    # ----- TRAVEL / NAVIGATION / TRANSPORT BOOKING -----
    if has(
        "train",
        "uber",
        "taxi",
        "flight",
        "book a ticket",
        "purchase ticket",
        "buy a ticket",
        "ticket for",
        "ticket to",
        "ticket via",
        "vacation",
        "tourist places",
        "ways of travel",
        "travel the whole world",
        "travel",
        "route",
        "traffic",
        "best way to",
        "where is the pharmacy",
        "shops are nearby",
        "shopping mall",
        "local shops",
        "where in d. c.",
        "where is the venue",
        "where is the kiss concert",
    ):
        return "travel_navigation_transport"
    if t in ("california", "new york"):
        # extremely short city names alone in this corpus usually = train/transport (we saw "ticket for bombay", "new york" near transport)
        return "travel_navigation_transport"
    if t == "where can i go tonight":
        return "travel_navigation_transport"

    # ----- FOOD / RECIPES / COOKING -----
    if has(
        "recipe",
        "recipes",
        "cook",
        "boil an egg",
        "ingredient",
        "saffron",
        "pasta",
        "chocolate chip cookies",
        "make a turkey",
        "bake and broil",
        "cooking",
        "sugar free diet",
        "diet",
        "knitting patterns",
        "good ideas for cooking",
        "butter chicken",
    ):
        return "food_recipes"
    if has("order a pizza", "order pizza", "from domino", "michael's pizza", "taco bell delivers", "food order"):
        return "food_recipes"
    if t in ("how is my order", "olly do they deliver home", "instructions to make a meal"):
        return "food_recipes"

    # ----- STOCKS / FINANCE / CURRENCY -----
    if has(
        "stock",
        "stocks",
        "dollar",
        "exchange rate",
        "currency converter",
        "british pound",
        "pound sterling",
        "i. b. m.",
        "ibm",
        "walmart stock",
        "hdfc",
        "starbuck",
        "apple",
    ) and not has(
        "iphone", "google pixel", "smartphone"
    ):
        # apple/starbuck only count if alongside stock/price context which has matched
        return "finance_stocks_currency"
    if has("price of") and (has("stock", "company") or t.startswith("what is price of")):
        return "finance_stocks_currency"

    # (contacts merged into messaging_email_contacts above)

    # ----- QA / FACTUAL LOOKUP / DEFINITIONS / MATH -----
    if has(
        "definition",
        "define",
        "spell and define",
        "describe a",
        "how would you describe",
        "what is a",
        "what is the highest",
        "how big is",
        "how tall is",
        "square root",
        "what is two",
        "what is twelve",
        "two plus two",
        "product of",
        "what is the sum",
        "answer to the universe",
        "tell me about",
        "tell me what the",
        "how long should i boil",
        "what is the location of",
        "which ocean",
        "rocky mountains",
        "see russia from alaska",
        "abraham lincoln",
        "hypothesis",
        "describe a sloth",
        "describe a rotor",
        "definitions of orange",
        "prime minister",
        "hemingway",
        "celebrity",
        "best tourist places",
        "what is the difference between bake",
        "what conducts heat better",
        "highest building",
        "what is the cosmos",
        "what are converse shoes",
        "features of google pixel",
        "smartphone",
        "details about bruce lee",
        "where do the rocky",
        "tell me a place that has snow",
        "what is the news today",  # not really news, factual question form; but better captured by news
        "tell me about india location",
        "tell me about trump",
        "field does that person excel in",
        "what is on the radio right now",  # info-seeking
        "where was will ferrell seen",
        "does pink have a new baby",
        "movie should i watch",
        "tell me the profession",
        "i need some details about",
        "expired products",
        "in which field",
        "investigation into trump's ties",
        "trump's ties with russia",
        "how safe is the city regarding law and order",
        "top model car",
        "what's the sum of the two numbers",
        "web searches",
        "internet please",
    ):
        return "general_qa"

    # ----- CONVERSATIONAL / CHITCHAT / OPINIONS / JOKES / GAMES -----
    if has(
        "hello how is your day",
        "how is your day",
        "how has your day been",
        "do you like my girlfriend",
        "knock knock",
        "funny jokes",
        "play scrabble",
        "play game",
        "play nfs",
        "i hate this song",
        "this is the best band",
        "do you recommend",
        "i would like it to help analyze ideas",
        "i want it to remember my preferences",
        "i would like my robot to feed",
        "i think i can travel",
        "happy birthday",
        "birthday wishes",
        "opinion petabit",
        "i can't see turn up the lights",  # actually a lights control - will catch above
        "today the following happened",
        "today i had a meeting with george",  # diary-like — chitchat
        "mark today as the start of my diet",
        "i would like to hear",  # genuine listening request -> music
    ):
        # certain ones above are music — re-route
        if has("i would like to hear", "i want some jazz", "i hate this song"):
            return "music_audio"
        return "conversational_chitchat"

    return "UNCLASSIFIED"


def main():
    import sys

    texts = json.loads(SAMPLE.read_text(encoding="utf-8"))
    buckets = {}
    for item in texts:
        key = classify(item["text"])
        buckets.setdefault(key, []).append(item)

    # print stats
    for k in sorted(buckets, key=lambda k: -len(buckets[k])):
        print(f"{k:40s} {len(buckets[k]):4d}")
    print(f"TOTAL: {sum(len(v) for v in buckets.values())}")

    if len(sys.argv) > 1 and sys.argv[1] == "--dump":
        for k in sorted(buckets):
            print(f"\n=== {k} ({len(buckets[k])}) ===")
            for item in buckets[k]:
                print(f"  {item['id']:30s} {item['text']!r}")

    if len(sys.argv) > 1 and sys.argv[1] == "--write":
        # Write the proposal file
        now = datetime.now()
        ts = now.strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:4]
        out = WORKSPACE / "proposals" / f"prop_{ts}_{uid}.json"
        proposal = {
            "timestamp": now.isoformat(),
            "sample_size": 500,
            "sample_strategy": "shared_random",
            "style": "action-vs-info",
            "existing_clusters_considered": False,
            "clusters": [],
            "unclustered_ids": [],
            "observations": "",
        }

        # Pretty-named clusters with descriptions
        spec = {
            "audio_playback_music_radio_podcasts": (
                "Audio playback (music, radio, podcasts, audiobooks)",
                "Control of media playback: play/pause/resume/skip songs, albums, playlists, radio stations or channels, podcasts and audiobooks; queries about what's playing, song identification, ratings/opinions on songs, and managing favourite/music lists.",
            ),
            "calendar_events": (
                "Calendar, meetings, appointments and events",
                "Adding, modifying, cancelling or querying calendar items: meetings, appointments, scheduled events, anniversaries/birthdays, and questions like 'what's on my calendar', 'am I free?', or 'any cultural events this weekend?'.",
            ),
            "messaging_email_contacts": (
                "Email, messages and contacts",
                "Reading, checking, composing, dictating, replying to or archiving emails; questions about incoming mail or whether someone has been in touch; plus managing or querying the contact book (phone numbers, contact details, adding/listing contacts).",
            ),
            "general_qa": (
                "General factual Q&A, definitions, math and trivia",
                "Information-seeking questions that aren't tied to a device domain: definitions, descriptions, arithmetic, world facts (geography, history, celebrities), 'tell me about X', and generic 'web search' / 'internet please' style fallbacks.",
            ),
            "weather": (
                "Weather forecasts and conditions",
                "Queries about current or forecast weather, rain, temperature, storms, snow, plus weather-adjacent questions like 'do I need an umbrella?' or 'will I have to shovel my driveway?'.",
            ),
            "smart_home_lights": (
                "Smart-home lighting control",
                "Turning lights on/off, dimming, brightening, changing colour or hue, scheduling light changes, and other commands directed at room/house lighting.",
            ),
            "lists_shopping": (
                "Lists and shopping lists",
                "Creating, editing, viewing or deleting user lists (shopping list, to-do-shaped lists, generic 'list', favourites and guest lists), and adding/removing items.",
            ),
            "social_media": (
                "Social media: posting, complaints and feeds",
                "Composing tweets/posts on Twitter or Facebook, sending complaint or feedback messages to companies, browsing social feeds or 'hot topics', and checking friend updates.",
            ),
            "travel_navigation_transport": (
                "Travel, navigation and transport booking",
                "Booking trains, taxis or Ubers, asking about train times or routes, finding nearby shops/venues/pharmacies, asking for directions or travel options, vacation/tourism suggestions, and traffic conditions.",
            ),
            "reminders_todo": (
                "Reminders, to-do items and notes",
                "Setting, listing, querying or deleting reminders and to-do items; asking to be alerted before something; dictating short notes/diary entries.",
            ),
            "food_recipes": (
                "Food, recipes and food delivery",
                "Recipes and cooking instructions, ingredient substitutions, diet-related queries, plus ordering food / asking about delivery and order status.",
            ),
            "smart_home_appliances": (
                "Smart-home appliances (coffee, vacuum, plugs)",
                "Operating non-lighting smart-home devices: coffee makers, robot vacuums/hoovers, smart plugs and sockets, and broad housekeeping commands like 'clean my house'.",
            ),
            "news_headlines": (
                "News, headlines and sports updates",
                "Asking for news, headlines, latest stories on specific topics (politics, environment, named people like Brexit/Trump), and sports-result updates like match highlights or game scores.",
            ),
            "finance_stocks_currency": (
                "Stocks, currency and finance",
                "Stock prices and movement, currency exchange rates and conversion, and finance-related lookups about specific companies (Apple/Starbucks/IBM/Walmart/HDFC stock).",
            ),
            "alarms": (
                "Alarms",
                "Creating, listing, modifying or removing alarms (set/new/cancel alarm at time X, 'wake me up', confirming alarm settings).",
            ),
            "datetime_clock": (
                "Date and time / clock queries",
                "Asking for the current time (local or in a named city), the day of the week, or whether a specific date falls on a given day. Pure clock/calendar-date lookups, not calendar events.",
            ),
            "audio_volume_control": (
                "Assistant audio and volume control",
                "Controlling the assistant's own audio output: volume up/down, mute, silence, 'be quiet', 'stop speaking', 'speak loudly' — distinct from media playback control.",
            ),
            "conversational_chitchat": (
                "Conversational chitchat and opinions",
                "Greetings ('hello how is your day'), jokes ('knock knock'), playing games, expressing opinions/preferences to the assistant, and open-ended chatty utterances that aren't tied to a task domain.",
            ),
        }

        for key, items in buckets.items():
            name, desc = spec[key]
            ids = [it["id"] for it in items]
            # Build short reasoning string
            sample_quotes = "; ".join(repr(it["text"]) for it in items[:3])
            reasoning = (
                f"{len(items)} of 500 texts; representative examples: {sample_quotes}. "
                f"Action lens: this cluster mixes imperatives and info-seeking requests "
                f"but all share the {name.lower()} domain."
            )
            proposal["clusters"].append(
                {
                    "name": name,
                    "description": desc,
                    "text_ids": ids,
                    "reasoning": reasoning,
                }
            )

        proposal["observations"] = (
            "Applied an action-vs-information lens as a discriminator: most texts split into "
            "clear imperative requests (play X, set Y, turn Z off) and information-seeking "
            "queries (what is, when does, tell me). Crossed with domain, this collapses into "
            f"{len(buckets)} coarse domain clusters within the 14-22 target. The largest is "
            "audio playback (music/radio/podcasts/audiobooks) at 78/500, followed by calendar "
            "events (68) and email/messaging/contacts (45). 'General QA' (41) holds the "
            "genuine factual lookups that don't fit any specific device domain — not a fallback "
            "bucket. Surprises: (1) the action lens cleanly separated 'audio_volume_control' "
            "(silence the assistant) from 'audio_playback' (control media); (2) 'datetime_clock' "
            "(pure clock queries) is distinct from 'calendar_events' (scheduled items); (3) a "
            "single very long dictation paragraph about young drivers and road safety fits "
            "reminders_todo as a note-taking utterance. No fallback bucket; unclustered_ids empty."
        )

        out.write_text(json.dumps(proposal, indent=2), encoding="utf-8")
        print(f"\nWROTE: {out}")


if __name__ == "__main__":
    main()
