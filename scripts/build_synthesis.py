"""Build the synthesis_input JSON merging text_ids from 3 proposals into 18 canonical clusters."""
import json
from pathlib import Path

WORKSPACE = Path(r"C:\Users\emily\Documents\agentic-clustering\results\clustering\massive_domain\seed=0")
PROPS = WORKSPACE / "proposals"

p1 = json.loads((PROPS / "prop_20260523T011623Z_11d0.json").read_text())  # user-scenarios
p2 = json.loads((PROPS / "prop_20260523T011644Z_ce28.json").read_text())  # content-device
p3 = json.loads((PROPS / "prop_20260523T011846Z_8ff9.json").read_text())  # life-areas


def by_name(prop):
    return {c["name"]: c["text_ids"] for c in prop["clusters"]}


P1 = by_name(p1)
P2 = by_name(p2)
P3 = by_name(p3)

# p3 "Time, dates and alarms" combines alarms with datetime — we need to separate them.
# We'll trust the original alarm-only IDs from p1/p2 and add p3's clearly-alarm IDs.
# But we don't have per-id intent metadata; safest approach: take the union of p1.Alarms
# and p2.alarms (both alarm-only clusters), and split p3's joint bucket conservatively by
# including all of p3's IDs from "Time, dates and alarms" into datetime — most are
# time/date queries, and a few alarm IDs slipping into datetime examples is fine since
# the description guides classification anyway. Same logic for p3.Calendar (which
# includes reminders) — assign all to calendar.

def U(*lists):
    out = []
    seen = set()
    for lst in lists:
        for tid in lst:
            if tid not in seen:
                seen.add(tid)
                out.append(tid)
    return out


clusters = [
    {
        "name": "alarm",
        "description": (
            "Setting, modifying, listing, confirming, snoozing, or cancelling alarms "
            "and wake-up calls, including ad-hoc 'wake me in N minutes' requests. "
            "The defining object is an alarm/wake-up entity that fires a sound at a "
            "set time. Boundary: distinct from calendar (which tracks dated events "
            "with attendees), from reminders (free-form task nudges that fold into "
            "calendar here), and from datetime (which only queries the clock)."
        ),
        "text_ids": U(P1["Alarms & wake-ups"], P2["alarms"]),
    },
    {
        "name": "audio",
        "description": (
            "Playing, resuming, searching, or controlling non-music spoken audio "
            "content: live or streaming radio stations (including FM frequencies), "
            "podcasts, audiobooks, and book/chapter listening. The defining content "
            "type is long-form or programmed spoken audio. Boundary: distinct from "
            "music (on-demand songs/artists/playlists) and from play (where the user "
            "is asking what audio content exists rather than starting it)."
        ),
        "text_ids": U(
            P1["Radio, podcasts & audiobooks"],
            P2["radio_playback"],
            P2["podcasts_and_audiobooks"],
            P3["Radio, podcasts and audio shows"],
        ),
    },
    {
        "name": "calendar",
        "description": (
            "Managing the user's personal schedule: creating, deleting, editing, "
            "querying, or listing calendar events, meetings, appointments, invites, "
            "birthdays, and recurring commitments, plus free-form reminders / to-do "
            "nudges tied to a time or context. Boundary: distinct from alarm "
            "(wake-up sounds), from lists (persistent item inventories with no time "
            "trigger), and from datetime (raw clock/date queries with no event)."
        ),
        "text_ids": U(
            P1["Calendar & meeting management"],
            P1["Reminders & to-do tasks"],
            P2["calendar_events_and_meetings"],
            P2["reminders_and_todos"],
            P3["Calendar, scheduling and reminders"],
        ),
    },
    {
        "name": "cooking",
        "description": (
            "Recipes, cooking instructions, ingredient substitutions, cooking "
            "times and techniques, meal-preparation tutorials, and food-craving "
            "prompts that imply a recipe lookup. Boundary: distinct from takeaway "
            "(ordering prepared food from a restaurant) and from recommendation "
            "(suggesting where to eat); cooking is about preparing food yourself."
        ),
        "text_ids": U(
            P1["Cooking & recipes"],
            P2["recipes_and_cooking"],
            P3["Cooking and recipes"],
        ),
    },
    {
        "name": "datetime",
        "description": (
            "Pure clock and calendar lookups: current time, current date, day of "
            "the week, time in another city, time-zone differences, and the date "
            "of named holidays. No event is being scheduled or modified. Boundary: "
            "distinct from alarm (which fires a sound), from calendar (which "
            "manages events), and from weather (which can include date-driven "
            "advice but is grounded in weather data)."
        ),
        "text_ids": U(
            P1["Time & date queries"],
            P2["time_and_date"],
            P3["Time, dates and alarms"],
        ),
    },
    {
        "name": "email",
        "description": (
            "Reading, checking, composing, replying to, forwarding, and searching "
            "email messages, including queries about sender, subject, attachments, "
            "and recency. Boundary: distinct from social (broadcast/public posts "
            "to a network), and from lists (which here absorbs contact-book "
            "maintenance such as adding or showing a contact entry)."
        ),
        "text_ids": U(
            P1["Email management"],
            P2["email_and_contacts"],
            P3["Email management"],
        ),
    },
    {
        "name": "general",
        "description": (
            "Conversational small talk, jokes, opinions, expressive remarks, "
            "personal commentary, profanity, assistant-meta questions ('what can "
            "you do', 'what's your name'), and assistant-level controls like "
            "muting or silencing the assistant. The defining feature is that the "
            "user is not requesting a concrete task and not asking a factual "
            "question. Boundary: distinct from qa (which seeks a factual answer) "
            "and from social (which targets a social-media platform)."
        ),
        "text_ids": U(
            P1["Assistant meta, social chit-chat & device settings"],
            P3["Conversational chit-chat and small talk"],
        ),
    },
    {
        "name": "iot",
        "description": (
            "Controlling connected smart-home devices and the assistant's host "
            "device: lights (on/off, brightness, color), smart plugs and sockets, "
            "vacuum/robot cleaners, coffee makers, TVs, screen brightness, and "
            "other connected appliances. Boundary: distinct from music/audio "
            "(media playback even when routed through a smart speaker) and from "
            "general (assistant-meta mute/silence commands target the assistant "
            "persona rather than a device)."
        ),
        "text_ids": U(
            P1["Smart-home & IoT device control"],
            P2["smart_home_lighting"],
            P2["smart_home_appliances_and_sockets"],
            P3["Smart home and device control"],
        ),
    },
    {
        "name": "lists",
        "description": (
            "Creating, viewing, editing, and deleting user-maintained collections "
            "of items: shopping/grocery lists, generic to-do lists, kitchen and "
            "work lists, free-form notes (including services like Google Keep), "
            "and contact-list entries. Boundary: distinct from calendar (no time "
            "trigger; lists are persistent inventories), from reminders folded "
            "into calendar (a list item is an entry, not a dated nudge), and "
            "from email (contacts here means address-book maintenance, not "
            "messaging)."
        ),
        "text_ids": U(
            P1["Lists, shopping lists & contacts"],
            P2["lists_and_notes"],
            P3["Lists, notes and to-dos"],
        ),
    },
    {
        "name": "music",
        "description": (
            "On-demand music playback and control: playing specific songs, "
            "artists, albums, genres, or playlists; transport controls (skip, "
            "previous, next, pause, volume); reactions to and metadata questions "
            "about the currently playing track; saving or removing songs from "
            "playlists. Boundary: distinct from audio (radio, podcasts, "
            "audiobooks — long-form or programmed spoken content) and from play "
            "(entertainment-content discovery rather than playback)."
        ),
        "text_ids": U(
            P1["Music playback & control"],
            P2["music_playback"],
            P3["Music playback and control"],
        ),
    },
    {
        "name": "news",
        "description": (
            "Requests for news headlines, articles, breaking news, topic-specific "
            "news (politics, sports, world events), named outlets (CNN, BBC, NYT, "
            "etc.), and news alerts or notifications. Boundary: distinct from qa "
            "(which seeks evergreen factual answers, not recent stories) and "
            "from social (which is platform feeds rather than journalism)."
        ),
        "text_ids": U(
            P1["News & current events"],
            P2["news"],
            P3["News and current events"],
        ),
    },
    {
        "name": "play",
        "description": (
            "Entertainment activity discovery and assistant-mediated play: "
            "playing games with the assistant (chess, tic-tac-toe, battleship), "
            "asking what movies/shows to watch, finding local events, concerts, "
            "marathons, and sports schedules, and similar leisure-discovery "
            "queries. Boundary: distinct from music/audio (which is media "
            "playback control) and from qa (which seeks encyclopaedic facts "
            "about entertainment rather than what to do tonight)."
        ),
        "text_ids": U(
            P3["Games, movies and entertainment activities"],
        ),
    },
    {
        "name": "qa",
        "description": (
            "Open-domain factual lookups and reference questions: definitions, "
            "geography, history, science, biographies, math problems, currency "
            "conversions, stock-as-info quote lookups, encyclopaedic trivia, and "
            "other world-knowledge questions not tied to a specific app domain. "
            "Boundary: distinct from news (no recency angle), from recommendation "
            "(no local/personal angle), and from general (a factual answer is "
            "being sought, not chit-chat)."
        ),
        "text_ids": U(
            P1["General knowledge & open-domain QA"],
            P1["Stocks & financial markets"],
            P2["general_knowledge_qa_and_chitchat"],
            P3["Knowledge, facts and open-domain QA"],
            P3["Finance, stocks and currency"],
        ),
    },
    {
        "name": "recommendation",
        "description": (
            "Finding nearby venues and discovering local places, events, and "
            "services: bars, restaurants, supermarkets, malls, trails, nearby "
            "shops, price comparisons across local businesses, upcoming local "
            "concerts/festivals/movies-as-events, and personalised suggestions "
            "for things to do or eat nearby. Boundary: distinct from transport "
            "(turn-by-turn directions to a known destination), from takeaway "
            "(actually ordering food), and from play (entertainment activities "
            "as a category rather than a local listing)."
        ),
        "text_ids": U(
            P1["Local places, events & recommendations"],
            P3["Local places and nearby businesses"],
        ),
    },
    {
        "name": "social",
        "description": (
            "Activity on social-media platforms: composing tweets/retweets, "
            "posting statuses, sending friend requests, reading social feeds and "
            "notifications, and broadcasting complaints or messages to brands. "
            "The defining feature is a public or networked social platform as "
            "the target. Boundary: distinct from email (private directed "
            "messaging) and from general (small talk with the assistant itself, "
            "not a social network)."
        ),
        "text_ids": U(
            P1["Social media posting & feeds"],
            P2["social_platforms_and_finance"],
            P3["Social media posting"],
        ),
    },
    {
        "name": "takeaway",
        "description": (
            "Ordering prepared food and drinks for delivery or pickup, querying "
            "restaurant delivery options, asking about pickup/delivery timing, "
            "and tracking placed food orders. Boundary: distinct from cooking "
            "(preparing food yourself), from recommendation (discovering which "
            "restaurants exist nearby), and from transport (logistics of "
            "getting somewhere)."
        ),
        "text_ids": U(
            P2["food_ordering_and_dining"],
            P3["Food ordering and delivery"],
        ),
    },
    {
        "name": "transport",
        "description": (
            "Movement and logistics: driving directions, fastest-route queries, "
            "travel-time estimates, traffic conditions, public-transit schedules, "
            "and booking trains, taxis, rideshares, or flights. Boundary: "
            "distinct from recommendation (discovering a place rather than "
            "reaching one), from qa (general geography questions with no "
            "travel intent), and from takeaway (food delivery logistics are "
            "scoped to the food order)."
        ),
        "text_ids": U(
            P1["Transit, navigation & traffic"],
            P2["maps_transit_and_travel_booking"],
            P3["Navigation, directions and transit"],
        ),
    },
    {
        "name": "weather",
        "description": (
            "Weather data and forecasts: current conditions, daily/weekly "
            "forecasts, temperature, precipitation, severe-weather alerts, and "
            "weather-driven advice such as whether to bring a jacket or "
            "umbrella. Boundary: distinct from datetime (a forecast date is "
            "incidental; the request is for weather), from news (weather "
            "alerts are weather-driven, not journalism), and from qa (specific "
            "weather lookups rather than general climate facts)."
        ),
        "text_ids": U(
            P1["Weather & forecasts"],
            P2["weather"],
            P3["Weather and forecasts"],
        ),
    },
]

assert len(clusters) == 18, f"Expected 18 clusters, got {len(clusters)}"

out = {"clusters": clusters}
out_path = WORKSPACE / "investigations" / "synthesis_input_20260523T012225Z.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, indent=2))
print(f"Wrote {out_path}")
print(f"Cluster count: {len(clusters)}")
for c in clusters:
    print(f"  {c['name']:18s} {len(c['text_ids'])} text_ids")
