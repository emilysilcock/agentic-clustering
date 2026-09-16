"""Build the speech-act / lexical-cue clustering proposal for the
proposer subagent task. Hand-curated cluster assignments based on
the 300 sampled massive_intent texts.
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

SAMPLE_PATH = Path(
    r"C:\Users\emily\.claude\projects\C--Users-emily-Documents-agentic-clustering"
    r"\cb91043a-21cb-41ff-bf06-e5ac47f057e7\tool-results\b8xe0dlu7.txt"
)
WORKSPACE = Path(
    r"C:\Users\emily\Documents\agentic-clustering\results\clustering"
    r"\massive_intent\seed=0_proposers_v2"
)

# --------------------------------------------------------------------------- #
# Cluster definitions. Each entry: (name, description, reasoning, [text_ids])
# Speech-act + content combo is the lens.
# --------------------------------------------------------------------------- #

CLUSTERS = [
    # ---------------- SMART-HOME LIGHTING (imperative + polite-can-you) ----- #
    (
        "Lights — explicit turn-off command",
        "Direct imperatives to switch a light or lights off (with or without "
        "location). Surface cues: 'turn off', 'lights off', 'please turn lights off'.",
        "Several texts share the bare 'turn off [the] lights [in X]' frame; speech-act "
        "is direct imperative.",
        [
            "massive_intent-test-000004",  # olly turn the lights off in the bedroom
            "massive_intent-test-001044",  # hey olly turn off the light please
            "massive_intent-test-000087",  # turn off the bathroom lights
            "massive_intent-test-000086",  # turn off the lights in the bathroom
            "massive_intent-test-000297",  # turn off all lights
            "massive_intent-test-000159",  # please turn lights off
        ],
    ),
    (
        "Lights — dim / lower brightness",
        "Requests to decrease light intensity. Surface cues: 'lower', 'turn down', "
        "'dim', 'lower light', 'I want a lower light'.",
        "Bundled under a single fine-grained intent: reduce-brightness, regardless "
        "of whether phrased as declarative ('I want a lower light') or imperative.",
        [
            "massive_intent-test-000129",  # hi can you please turn lower the lights
            "massive_intent-test-000547",  # i want a lower light
            "massive_intent-test-000906",  # turn my lights down to a lower level of brightness
            "massive_intent-test-000569",  # turn down the lights to medium
        ],
    ),
    (
        "Lights — brighten / raise brightness",
        "Requests to increase light intensity. Surface cues: 'turn up', 'brighten', "
        "'I can't see turn up the lights'.",
        "Symmetric counterpart to the dim cluster; very few but a clearly distinct "
        "intent.",
        [
            "massive_intent-test-000374",  # i can't see turn up the lights
            "massive_intent-test-000971",  # brighten of lights in living room
        ],
    ),
    (
        "Lights — color / hue change",
        "Requests to change light color or warmth. Surface cues: 'amber', 'blue', "
        "'red ish', explicit color words.",
        "Color-change is a discrete intent distinct from on/off/brightness, "
        "consistently triggered by color adjectives.",
        [
            "massive_intent-test-000158",  # make the house lights amber at six p. m.
            "massive_intent-test-000738",  # house can you make all the lights in the house blue
            "massive_intent-test-000600",  # i want the light to be a little more red ish
        ],
    ),
    # ---------------- SMART-HOME OTHER DEVICES ------------------------------ #
    (
        "Smart plug / device power-on",
        "Turn a non-light smart device on. Surface cues: 'power up the plug', "
        "'turn program on xmtune' (here interpreted as device-power).",
        "Distinct from lights; the verb 'power up' / 'turn on' targets an "
        "appliance or socket.",
        [
            "massive_intent-test-000085",  # power up the plug socket one
            "massive_intent-test-001645",  # turn program on xmtune
        ],
    ),
    (
        "Coffee machine — make coffee",
        "Polite-imperative or 'I-want' requests to make a coffee, with or without "
        "modifiers. Surface cues: 'make me a coffee', 'please make coffee', "
        "'set my coffee machine'.",
        "Bundle of make-coffee variants; speech-acts vary (please-X, can-you-X, "
        "bare imperative) but content is identical.",
        [
            "massive_intent-test-000313",  # please make me coffee without sweetener
            "massive_intent-test-000606",  # set my coffee machine
            "massive_intent-test-000530",  # make me a cup of coffee with salted carmel
            "massive_intent-test-001020",  # make a coffee please
            "massive_intent-test-000792",  # can you make some coffee
            "massive_intent-test-000480",  # please get the coffee machine to make me some coffee
            "massive_intent-test-000578",  # can you set my coffee machine to make me coffee at seven am
        ],
    ),
    # ---------------- AUDIO / VOLUME ---------------------------------------- #
    (
        "Volume — change / lower",
        "Volume adjustment requests. Surface cues: 'turn the volume down', "
        "'change the volume', 'speaker volume increase', 'silence speakers'.",
        "Grouped imperatives whose object is volume/speaker level. Includes the "
        "ambiguous 'change the volume at' fragment.",
        [
            "massive_intent-test-000974",  # can you change the volume at
            "massive_intent-test-000276",  # please turn the volume down
            "massive_intent-test-000830",  # speaker volume increase
            "massive_intent-test-000470",  # silence speakers
            "massive_intent-test-000001",  # quiet
        ],
    ),
    # ---------------- PLAY MUSIC ------------------------------------------- #
    (
        "Play — specific song/track",
        "Direct play imperatives naming a specific song/track. Surface cue: "
        "'play [SONG] by [ARTIST]', 'play track N from...'.",
        "Most common 'play X' frame; differentiated from artist / playlist / "
        "genre by the named-song complement.",
        [
            "massive_intent-test-000542",  # play poker face by lady gaga
            "massive_intent-test-000598",  # please play shake it off by taylor swift
            "massive_intent-test-000531",  # play track one from my david bowie playlist
            "massive_intent-test-000573",  # play the most popular elton john song
            "massive_intent-test-000898",  # please play the latest song from the album abbas
            "massive_intent-test-000141",  # play bilando
        ],
    ),
    (
        "Play — artist / album",
        "Play-by-artist or by-album requests without naming a specific track. "
        "Surface cues: 'play the beatles', 'open songs from major lazer', "
        "'open bad religion folder', 'play something from keane's album'.",
        "Distinct intent: play an artist's body of work or a whole album.",
        [
            "massive_intent-test-000334",  # play the beatles
            "massive_intent-test-000103",  # open songs from major lazer
            "massive_intent-test-000089",  # open bad religion folder
            "massive_intent-test-000525",  # play something from keane's hopes and fears album
            "massive_intent-test-000224",  # i like the songs of yeshudas please play it
        ],
    ),
    (
        "Play — genre / mood playlist",
        "Requests to play a genre or themed playlist. Surface cues: 'play my rock "
        "playlist', 'play nineties hip hop', 'play some christian music', "
        "'play only music of pop mix', 'play only my list'.",
        "Speech-act is play-imperative; content slot is genre/mood/playlist label.",
        [
            "massive_intent-test-000036",  # play my rock playlist
            "massive_intent-test-001638",  # play nineties hip hop
            "massive_intent-test-000157",  # play some christian music
            "massive_intent-test-000918",  # please play only music of pop mix
            "massive_intent-test-000594",  # play only my list
        ],
    ),
    (
        "Music — generic / declarative 'I want music'",
        "Vague music-on commands or first-person declaratives that imply music "
        "playback. Surface cues: bare 'music', 'I feel like jazz right now', "
        "'hey olly I like music by sigur ros', 'play next' (continuation).",
        "Captures declarative + bare-noun music intents that don't name a "
        "track/artist/playlist; speech-act is hint or continuation, not "
        "command-with-target.",
        [
            "massive_intent-test-000942",  # music
            "massive_intent-test-000044",  # i feel like jazz right now how about you
            "massive_intent-test-000791",  # hey olly i like music by sigur ros
            "massive_intent-test-001968",  # play next
            "massive_intent-test-001620",  # play
        ],
    ),
    (
        "Music — feedback / reaction / replay current",
        "First-person declaratives about the currently-playing song: reactions "
        "('I love this song', 'nice lyrics'), like-logging ('remember how I fell "
        "about this song'), and replay ('I want to play the song again').",
        "Speech-act is reaction/affect or replay-request scoped to the current "
        "track — the assistant logs like/dislike or restarts playback.",
        [
            "massive_intent-test-000508",  # i love this song
            "massive_intent-test-000696",  # nice lyrics
            "massive_intent-test-000827",  # remember how i fell about this song
            "massive_intent-test-000509",  # i want to play the song again
        ],
    ),
    (
        "Music — identify current track",
        "Wh-questions about the song that is currently playing. Surface cues: "
        "'who sings the song that I am listening to', 'title of song'.",
        "Distinct factual question scoped to currently-playing audio.",
        [
            "massive_intent-test-000663",  # who sings the song that i am listening to right now
            "massive_intent-test-000914",  # title of song
        ],
    ),
    # ---------------- PODCAST / AUDIOBOOK / RADIO --------------------------- #
    (
        "Podcast playback",
        "Requests to play a podcast or its next episode. Surface cues: 'play my "
        "favorite podcast', 'I want to listen to podcast', 'play next episode "
        "of podcast'.",
        "Separated from music: podcast as a content type has its own listener "
        "intent.",
        [
            "massive_intent-test-002011",  # play my favorite podcast please
            "massive_intent-test-001991",  # i want to listen to podcast
            "massive_intent-test-001985",  # play next episode of podcast
        ],
    ),
    (
        "Audiobook playback",
        "Requests to start or continue an audiobook. Surface cues: 'play me a "
        "random audio book', 'keep reading the audiobook', 'put on the giver'.",
        "Distinct content type from music/podcast; speech-acts include both "
        "start-new and continue.",
        [
            "massive_intent-test-001713",  # play me a random audio book that has to do with love
            "massive_intent-test-001692",  # keep reading the audiobook to me
            "massive_intent-test-001674",  # can you put on the giver
            "massive_intent-test-001803",  # play for me the game harry potter and the chamber of secrets
        ],
    ),
    (
        "Radio — tune to station/frequency",
        "Imperatives to start radio or tune to a specific station/frequency. "
        "Surface cues: 'tune to', 'change the station to', 'start bbc radio', "
        "'play station gx in the radio', 'open the radio app'.",
        "All texts target the radio modality with station-selection semantics.",
        [
            "massive_intent-test-001596",  # start b. b. c. radio
            "massive_intent-test-001613",  # tune to classic hits
            "massive_intent-test-001656",  # change the station to eighty two point four
            "massive_intent-test-001606",  # tune in to eight hundred and ninety seven f. m.
            "massive_intent-test-001651",  # play station gx in the radio
            "massive_intent-test-001605",  # open the radio app
            "massive_intent-test-001626",  # radio channels
        ],
    ),
    # ---------------- WEATHER / FORECAST ----------------------------------- #
    (
        "Weather — current / today's conditions",
        "Wh-questions about current or today's weather. Surface cues: 'how is "
        "the weather', \"what's the weather like\", with locative.",
        "Current-weather is a recurrent narrow intent distinct from forecast / "
        "rain-specific / clothing-advice.",
        [
            "massive_intent-test-000772",  # how is the weather in san francisco
            "massive_intent-test-000041",  # how is the weather like today
            "massive_intent-test-000166",  # what's the weather like right now in new york
            "massive_intent-test-000235",  # can i please have the weather for tomorrow here in costa mesa
        ],
    ),
    (
        "Weather — multi-day forecast",
        "Requests for the week/multi-day weather forecast. Surface cues: 'forecast "
        "for the week', 'seven day forecast', 'will it rain this week'.",
        "Time horizon distinguishes from current-conditions cluster.",
        [
            "massive_intent-test-000556",  # what is the forecast for the week
            "massive_intent-test-001000",  # display the seven day forecast for this week
            "massive_intent-test-000261",  # will it rain this week
            "massive_intent-test-000730",  # is there any rain in the forecast for the next week
        ],
    ),
    (
        "Weather — precipitation / storm specific",
        "Polar questions about rain, snow, storms, sticky/humid nights. Surface "
        "cues: 'is there snow', 'are storms likely', 'sticky night', 'when will "
        "it rain'.",
        "Polar-Q speech-act on a precipitation predicate; behaves differently "
        "from open-ended weather.",
        [
            "massive_intent-test-000593",  # is there snow in the forecast
            "massive_intent-test-001052",  # are storms likely today
            "massive_intent-test-001038",  # will it be a sticky night
            "massive_intent-test-000279",  # when is the next time it will rain
        ],
    ),
    (
        "Weather — actionable clothing/activity advice",
        "Decision-seeking questions about how to dress/act given the weather. "
        "Surface cues: 'what jacket should I wear', 'should I wear raincoat', "
        "'can I mow the grass', 'change car tires to snow tires'.",
        "Distinct because the speech-act is decision-advice, not a fact request.",
        [
            "massive_intent-test-000771",  # what jacket should i wear
            "massive_intent-test-001006",  # should i wear raincoat before getting out
            "massive_intent-test-000726",  # am i going to be able to mow the grass this evening
            "massive_intent-test-001002",  # shall i change my car tires to snow tires soon
        ],
    ),
    # ---------------- ALARMS / WAKE / TIMERS -------------------------------- #
    (
        "Alarm — set new",
        "Requests to create a new alarm. Surface cues: 'create an alarm', 'I want "
        "an alarm', 'wake me up in X', 'set an alarm when'.",
        "All instantiate the create-alarm intent; speech-acts include "
        "imperatives and I-want declaratives.",
        [
            "massive_intent-test-000835",  # wake me up in thirty minutes
            "massive_intent-test-000630",  # i want an alarm for three today
            "massive_intent-test-001057",  # create an alarm for today at ten am
            "massive_intent-test-000694",  # i want to set an alarm when i'm driving home tomorrow to remember to stop
        ],
    ),
    (
        "Alarm — cancel / delete",
        "Cancel an existing alarm. Surface cue: 'cancel my seven am alarm'.",
        "Single sample instance but the cancel-alarm intent is distinct from "
        "set/query.",
        [
            "massive_intent-test-000011",  # cancel my seven am alarm
        ],
    ),
    (
        "Alarm — query status / settings",
        "Polar questions about an alarm's existence/state. Surface cues: 'check "
        "if alarm is set', 'check if default alarm is set', 'confirm my alarm "
        "settings'.",
        "Speech-act is check/confirm — querying, not commanding.",
        [
            "massive_intent-test-001017",  # check if alarm is set for six am
            "massive_intent-test-000889",  # check if default alarm is set
            "massive_intent-test-000145",  # confirm my alarm settings
        ],
    ),
    # ---------------- TIME / DATE ------------------------------------------ #
    (
        "Time-of-day — query (local / timezone)",
        "Wh-questions about the current time, possibly in another city or "
        "specifying a future moment. Surface cues: 'what is the time in X', "
        "'present time in', 'tell me when it is five p.m.', 'in how many hours "
        "will it be midnight'.",
        "Time-as-information requests, including future-time triggers.",
        [
            "massive_intent-test-000756",  # what is the time in china
            "massive_intent-test-000795",  # what is the time in chicago
            "massive_intent-test-000208",  # present time in new york
            "massive_intent-test-000798",  # tell me when it is five p. m.
            "massive_intent-test-002320",  # can you tell me the time it is
            "massive_intent-test-000751",  # in how many hours will it be midnight in london england
        ],
    ),
    (
        "Date / day-of-week query",
        "Questions about today's or a specific date's day-of-week, including "
        "'is today X', 'when is X holiday', 'how many Saturdays in March'.",
        "Calendar-fact lookups (not weather or schedule); distinguished from "
        "personal calendar items.",
        [
            "massive_intent-test-000017",  # what date is it today
            "massive_intent-test-000068",  # what day of the week is first april
            "massive_intent-test-000692",  # what day is the fifth
            "massive_intent-test-001506",  # is today saint patricks day
            "massive_intent-test-000242",  # when is easter in the year two thousand and eighteen
            "massive_intent-test-000778",  # how many saturdays are in march
        ],
    ),
    # ---------------- CALENDAR / EVENTS ------------------------------------ #
    (
        "Calendar — add event/meeting",
        "Imperative requests to add a meeting, event, or appointment to the "
        "calendar. Surface cues: 'add', 'schedule', 'set an event', 'mark that I "
        "have a meeting'.",
        "All map to create-calendar-event; includes assistant-name vocatives.",
        [
            "massive_intent-test-001476",  # add jane's birthday party for tomorrow two p. m. at one hundred and twenty three main on my calendar
            "massive_intent-test-001464",  # add a meeting at the office with brian for three p. m. on tuesday
            "massive_intent-test-001409",  # schedule a sales meeting for wednesday at eleven am
            "massive_intent-test-001201",  # set an event for friday
            "massive_intent-test-001338",  # will you mark that i have a meeting with tom at three tomorrow
            "massive_intent-test-001240",  # new event
            "massive_intent-test-001326",  # buddy adding event in your calendar
            "massive_intent-test-001317",  # i want a meeting till three o'clock
        ],
    ),
    (
        "Calendar — remove single event",
        "Delete one specific event from the calendar. Surface cues: 'remove X', "
        "'delete', 'cancel my plans to', 'erase'.",
        "Targeted single-event removal; speech-act is imperative.",
        [
            "massive_intent-test-001881",  # remove my dentist's appointment from today's schedule
            "massive_intent-test-001377",  # alexa remove dinner with mike from my calendar
            "massive_intent-test-001518",  # delete the shopping trip i have scheduled march twenty third
            "massive_intent-test-001366",  # delete event from calendar
            "massive_intent-test-001191",  # cancel next meeting
            "massive_intent-test-001254",  # cancel the tomorrow's meeting from my calendar
            "massive_intent-test-001589",  # cancel my plans to pick up my parents from the airport
            "massive_intent-test-001529",  # clear my next activity
        ],
    ),
    (
        "Calendar — clear all events",
        "Wipe everything from the calendar over a range. Surface cues: 'clear "
        "everything off', 'erase all events'.",
        "Distinguished from single-event delete by 'all/everything' scope.",
        [
            "massive_intent-test-001430",  # clear everything off my calendar for the rest of the year
            "massive_intent-test-001501",  # erase all events from my calendar
        ],
    ),
    (
        "Calendar — query upcoming events",
        "Wh- and polar-questions about scheduled events or schedule overview. "
        "Surface cues: 'do I have appointments', 'what does my schedule look "
        "like', 'are there any meetings', 'what days do I have booked', 'what "
        "are my plans for May'.",
        "Read-side intent on calendar; speech-acts split between polar and wh.",
        [
            "massive_intent-test-001306",  # do i have appointments today
            "massive_intent-test-001477",  # what does my schedule look like today
            "massive_intent-test-001580",  # what days do i have booked
            "massive_intent-test-001473",  # what are my plans for the month of may
            "massive_intent-test-001227",  # are there any meetings set for next wednesday
            "massive_intent-test-001206",  # show me my calendar event this friday afternoon
            "massive_intent-test-001541",  # give me today's calendar events after six p. m.
            "massive_intent-test-001315",  # do i have anything going on
            "massive_intent-test-001292",  # what's for today
            "massive_intent-test-001459",  # i want to know more about this event
        ],
    ),
    # ---------------- REMINDERS -------------------------------------------- #
    (
        "Reminder — set new",
        "Set a reminder for a future task/event. Surface cues: 'remind me to', "
        "'set a reminder', 'set repeating reminder', 'send me a notification "
        "every year'.",
        "All map to create-reminder; includes one-off and repeating.",
        [
            "massive_intent-test-001217",  # remind me to pick up linda at five in the evening on seventh
            "massive_intent-test-001427",  # please remind me to go to the post office
            "massive_intent-test-001410",  # set a reminder in one hour for my bread to bake
            "massive_intent-test-001537",  # set repeating reminder for every sunday
            "massive_intent-test-001496",  # can you remind me to order the turkey three weeks before thanksgiving
            "massive_intent-test-001321",  # remind me about the meeting tomorrow one hour before
            "massive_intent-test-001553",  # just set me reminders about pending bill payments
            "massive_intent-test-001268",  # it's my mother's birthday today send me a notification every year on this date
            "massive_intent-test-001510",  # set a repeating reminder alarm for the facebook event i have to attend on seventh april
            "massive_intent-test-000271",  # once a new topic on politics comes up alert me
        ],
    ),
    (
        "Reminder — query / count",
        "Questions about existing reminders. Surface cues: 'how many reminders "
        "do I have', 'did I tell you to remind me something', 'remind me about "
        "my schedule for the afternoon' (read-back framing).",
        "Read-side reminders intent — counts and recall.",
        [
            "massive_intent-test-001197",  # how many reminders do i have
            "massive_intent-test-001387",  # did i tell you to remind me something
            "massive_intent-test-001190",  # remind me about my schedule for the afternoon
        ],
    ),
    # ---------------- LISTS (grocery, generic) ----------------------------- #
    (
        "List — add item",
        "Add an item to a list (grocery / shopping / generic). Surface cues: "
        "'add an item', 'please update my grocery list with', 'add new item to "
        "list'.",
        "Speech-act: imperative add; object: list item.",
        [
            "massive_intent-test-001872",  # can you please add an item to my grocery list
            "massive_intent-test-001852",  # please update my grocery list with one gallon of two percent milk
            "massive_intent-test-001930",  # add new item to list
            "massive_intent-test-001908",  # please add list of things to buy for party
        ],
    ),
    (
        "List — remove item / delete list",
        "Remove an item from a list or remove a whole list. Surface cues: "
        "'scratch that one', 'remove my list of favorite albums'.",
        "Symmetric to add-item; distinct intent.",
        [
            "massive_intent-test-001919",  # scratch that one from the list
            "massive_intent-test-001913",  # please remove my list of favorite albums
        ],
    ),
    (
        "List — create new",
        "Create a brand-new list. Surface cues: 'I need to make a list', 'please "
        "create new list'.",
        "Speech-act: imperative create; distinguished from add-item.",
        [
            "massive_intent-test-001885",  # i need to make a list
            "massive_intent-test-001946",  # please create new list
        ],
    ),
    (
        "List — read / show contents",
        "Show the contents or all available lists. Surface cues: 'show me the "
        "contents', 'give me all my lists', 'read my grocery list', 'any "
        "special events on my list', 'inform me on the items on the list'.",
        "Read-side list intent.",
        [
            "massive_intent-test-001886",  # read my grocery list
            "massive_intent-test-001848",  # give me all my lists
            "massive_intent-test-001850",  # show me the contents of the list
            "massive_intent-test-001951",  # give me all available lists
            "massive_intent-test-001867",  # inform me on the items on the list
            "massive_intent-test-001822",  # any special events on my list
        ],
    ),
    # ---------------- EMAIL ------------------------------------------------- #
    (
        "Email — compose / send new",
        "Compose and send a new email. Surface cues: 'send an email to', 'I "
        "need an email to be sent', 'write an email to', 'send a mail to'.",
        "Outbound new-email intent.",
        [
            "massive_intent-test-002859",  # i need an email to be sent to comcastcom about my service issues
            "massive_intent-test-002846",  # can you write an email to chelsea
            "massive_intent-test-002909",  # send an email to jerry ask what time will he be home tonight
            "massive_intent-test-002839",  # please send a mail to my friend divya how are you
            "massive_intent-test-002724",  # please send to new email address listed
        ],
    ),
    (
        "Email — reply to existing",
        "Reply to an already-received email. Surface cues: 'reply to', 'pull up "
        "X's email and write...'.",
        "Distinct: reply-context vs. compose-new.",
        [
            "massive_intent-test-002807",  # i want to reply to bob smith's email
            "massive_intent-test-002787",  # pull up kate's email and write that i will let her know
        ],
    ),
    (
        "Email — check / query inbox",
        "Check inbox or ask about new emails (with optional sender filter). "
        "Surface cues: 'check my emails for X', 'have I gotten any new email', "
        "'has X emailed me', 'do I have any incoming emails'.",
        "Read-side inbox query intent.",
        [
            "massive_intent-test-002747",  # check my emails for something from name
            "massive_intent-test-002785",  # check email about my job
            "massive_intent-test-002727",  # check to see if i have any new emails from my dad
            "massive_intent-test-002862",  # check mail for anything from shelly
            "massive_intent-test-002879",  # please check my emails and notify me if exists any new email
            "massive_intent-test-002820",  # have i gotten any new email
            "massive_intent-test-002866",  # do i have any incoming emails
            "massive_intent-test-002933",  # has jane doe emailed me
            "massive_intent-test-002811",  # has tom emailed me about ammunition
        ],
    ),
    (
        "Email — show / read content",
        "Display or extract content from a specific email. Surface cues: "
        "'display recent email from john', 'what is the interview time from "
        "yesterday's email'.",
        "Distinct from inbox-check: targets the body content.",
        [
            "massive_intent-test-002764",  # display recent email from john
            "massive_intent-test-002889",  # what is the interview time from yesterday's email
        ],
    ),
    # ---------------- SOCIAL MEDIA ------------------------------------------ #
    (
        "Social media — post / tweet content",
        "Post to twitter/facebook/instagram. Surface cues: 'tweet to', 'post on "
        "instagram', 'post my business on twitter', 'send a twitter complaint'.",
        "Outbound social-post intent; includes complaint-tweets which are still "
        "post-content speech-acts.",
        [
            "massive_intent-test-002651",  # send a twitter complaint
            "massive_intent-test-002689",  # tweet to apple about non receipt of iphone i sent for repairs last week
            "massive_intent-test-002698",  # tweet a disgusted face about mcdonalds
            "massive_intent-test-002679",  # please tweet a complaint to comcast for bad customer service
            "massive_intent-test-002685",  # tweet to google costumer service my new nexus phone stop working help please
            "massive_intent-test-002597",  # post my business on twitter
            "massive_intent-test-002614",  # my trip to goa photo post it on instagram
            "massive_intent-test-002596",  # can you upload my latest selfie in my facebook account
        ],
    ),
    (
        "Social media — query feed",
        "Read social-media feed. Surface cues: 'is there any post from my "
        "friend mike in facebook', bare 'facebook'.",
        "Read-side social intent.",
        [
            "massive_intent-test-002613",  # is there any post from my friend mike in facebook
            "massive_intent-test-002620",  # facebook
        ],
    ),
    # ---------------- NEWS -------------------------------------------------- #
    (
        "News — general headlines",
        "Generic news/headlines query. Surface cues: 'what are the latest news "
        "headlines', 'get me the popular news from BBC', 'any breaking news', "
        "'new updates', 'plant based news', 'refer local current events', "
        "'is there any news olly'.",
        "Generic-news intent regardless of source modifier.",
        [
            "massive_intent-test-000090",  # what are the latest news headlines
            "massive_intent-test-000923",  # get me the popular news from b. b. c.
            "massive_intent-test-001022",  # any breaking news from the huffington post
            "massive_intent-test-000832",  # is there any news olly
            "massive_intent-test-002606",  # new updates
            "massive_intent-test-001113",  # plant based news
            "massive_intent-test-002107",  # refer local current events
            "massive_intent-test-000282",  # so clean news in past six hours is what
            "massive_intent-test-000767",  # what are the trending articles on the new york times
        ],
    ),
    (
        "News — specific topic / person",
        "News query scoped to a topic, person, or holiday. Surface cues: 'give "
        "me news on president trump', 'latest news on international women's "
        "day', bare 'trump'.",
        "Topic-scoped news, distinguished from general headlines by the "
        "subject-matter slot.",
        [
            "massive_intent-test-000299",  # give me news on president trump
            "massive_intent-test-000202",  # latest new on international women's day
            "massive_intent-test-001134",  # trump
        ],
    ),
    # ---------------- STOCKS / FINANCE ------------------------------------- #
    (
        "Stocks — price / performance query",
        "Wh- and polar-questions about a stock's price or trend. Surface cues: "
        "'how is X doing', 'how have X shares done', 'tell me the current price "
        "of', 'were the stocks rising or declining', 'open stock price for X'.",
        "Stock-information intent; speech-acts are mostly wh-questions.",
        [
            "massive_intent-test-002593",  # how is i. b. m. doing
            "massive_intent-test-002565",  # how have megatel shares done last week
            "massive_intent-test-002366",  # tell me the current price of exxon mobil stock
            "massive_intent-test-002562",  # were the stocks rising or declining
            "massive_intent-test-002493",  # open stock price for name
        ],
    ),
    (
        "Currency / exchange rate query",
        "Wh-questions about exchange rates. Surface cue: 'what is the exchange "
        "rate between X and Y'.",
        "Narrow finance intent distinct from stocks.",
        [
            "massive_intent-test-002374",  # what is the exchange rate between us and mexico
            "massive_intent-test-002379",  # what is the exchange rate for mexico money
        ],
    ),
    # ---------------- TRAVEL / TRANSPORT ----------------------------------- #
    (
        "Train — find/book ticket",
        "Find or book a train ticket. Surface cues: 'find a train ticket', 'give "
        "me the list of available train tickets', 'find me the cheapest train "
        "ticket', 'can you book a train ticket', 'ticket for delhi'.",
        "Train-booking intent; speech-acts include can-you and bare imperative.",
        [
            "massive_intent-test-002167",  # find a train ticket to philadelphia
            "massive_intent-test-002143",  # give me the list of available train tickets from edinburgh to leeds
            "massive_intent-test-002174",  # find me the cheapest train ticket to spain
            "massive_intent-test-002116",  # can you book a train ticket
            "massive_intent-test-002138",  # ticket for delhi
        ],
    ),
    (
        "Train — schedule query",
        "Ask when trains run / next train leaves. Surface cues: 'tell me the "
        "times the train leaves', 'when does the next train leave'.",
        "Train-info (schedule) intent, separate from booking.",
        [
            "massive_intent-test-002164",  # can you tell me the times the train leaves for chicago
            "massive_intent-test-002156",  # when does the next train traveling the city leave here
        ],
    ),
    (
        "Taxi / rideshare booking",
        "Book a taxi/uber/lyft. Surface cue: 'call an uberpool to get me at X'.",
        "Distinct rideshare intent.",
        [
            "massive_intent-test-002195",  # call an uberpool to get me at long island bar
        ],
    ),
    # ---------------- FOOD / RECIPES / ORDER -------------------------------- #
    (
        "Food — order delivery",
        "Order food from a restaurant. Surface cues: 'I want to order a pizza "
        "from X', 'hey order two wings'.",
        "Speech-act: imperative or I-want declarative on food-order frame.",
        [
            "massive_intent-test-000410",  # i want to order a pizza from michael's pizza
            "massive_intent-test-001062",  # hey order two wings with french fries from the chinese food store
            "massive_intent-test-001760",  # i would like a cheeseburger
        ],
    ),
    (
        "Food — order status query",
        "Polar/wh-questions about delivery status. Surface cues: 'when will my "
        "chinese food be delivered', 'is the last order ready', 'does X "
        "deliver'.",
        "Order-status intent; distinct from order-placement.",
        [
            "massive_intent-test-000729",  # when will my chinese food be delivered
            "massive_intent-test-000451",  # is the last order is ready
            "massive_intent-test-000551",  # olly does shibaru sushi deliver
        ],
    ),
    (
        "Recipe — find / lookup",
        "Find or look up a recipe. Surface cues: 'find me a recipe', 'find a "
        "recipe for X', 'find the recipe for X in app', 'show me a video on "
        "cooking X'.",
        "Recipe-find intent; mostly imperatives.",
        [
            "massive_intent-test-001721",  # find me a chocolate cake recipe
            "massive_intent-test-001754",  # find a recipe for dinner tonight
            "massive_intent-test-001780",  # find the recipe for sambar in cookingforu application
            "massive_intent-test-001726",  # show me a video on cooking fried chicken
        ],
    ),
    (
        "Cooking — technique / time / substitution Q",
        "Wh- and 'how long' questions about cooking technique, time, or "
        "ingredient substitutions. Surface cues: 'what is the best way to', "
        "'how long should I boil', 'what ingredient can be used instead of', "
        "'baking times for'.",
        "Speech-act: question on cooking knowledge — distinct from recipe-find.",
        [
            "massive_intent-test-001729",  # what is the best way to cook pasta al dente
            "massive_intent-test-001752",  # baking times for chicken in the oven
            "massive_intent-test-001732",  # what ingredient can be used instead of saffron
            "massive_intent-test-001720",  # olly how long should i boil the egg
        ],
    ),
    # ---------------- LOCAL / EVENTS / RECOMMENDATIONS --------------------- #
    (
        "Local events / things to do query",
        "Ask what's happening or where to go locally. Surface cues: 'show me "
        "nearby musical events', 'show me painting exhibition in bay area', "
        "'give me the list of circus shows', 'anything interesting going on in "
        "the bay area', 'where can I go tonight'.",
        "Local-events recommendation intent.",
        [
            "massive_intent-test-002108",  # show me nearby musical events
            "massive_intent-test-002051",  # show me painting exhibition in bay area
            "massive_intent-test-002033",  # give me the list of circus shows going on in the city right now
            "massive_intent-test-002071",  # anything interesting going on in the bay area
            "massive_intent-test-002068",  # where can i go tonight
        ],
    ),
    (
        "Bar / venue recommendation",
        "Ask for nearby bars / drink prices. Surface cues: 'which bar has the "
        "most affordable drinks near sixth street', 'olly i'm looking for a "
        "bar do you know a good one'.",
        "Bar/venue-recommendation intent.",
        [
            "massive_intent-test-002063",  # which bar has the most affordable drinks near sixth street
            "massive_intent-test-002043",  # olly i'm looking for a bar do you know a good one
        ],
    ),
    # ---------------- FACTUAL QUESTION ANSWERING --------------------------- #
    (
        "Person — biography / age / net worth",
        "Wh-questions about celebrities/people: birthday, age, net worth, "
        "address. Surface cues: 'how old is X', \"what is X's birthday\", "
        "\"what is X's networth\", 'tell me X's address', 'where does X live'.",
        "Person-factoid speech-act.",
        [
            "massive_intent-test-002343",  # what is arnold schwarzenegger's birthday
            "massive_intent-test-002579",  # how old is clint eastwood
            "massive_intent-test-002308",  # how old is j. k. rowling
            "massive_intent-test-002416",  # how tall is hulk hogan
            "massive_intent-test-002458",  # what is leonardo dicaprio's networth
            "massive_intent-test-002537",  # where does sophia vergara live
            "massive_intent-test-002250",  # tell me billy crytals address
            "massive_intent-test-002265",  # check celebrity where abouts
            "massive_intent-test-002280",  # what was george eliot's first november el
            "massive_intent-test-002430",  # show bio of rihana
            "massive_intent-test-002390",  # can you tell me about wayne gretszky
        ],
    ),
    (
        "Open-ended 'tell me about X' factual lookup",
        "Tell-me-about a topic, including describe / why / how questions. "
        "Surface cues: 'tell me all about X', 'describe X', 'why is the earth "
        "round', 'how would you describe X'.",
        "Encyclopedic open-ended fact request; speech-act 'tell me about'.",
        [
            "massive_intent-test-002259",  # tell me all about hurricane
            "massive_intent-test-002427",  # describe rock sand
            "massive_intent-test-002323",  # why is the earth round
            "massive_intent-test-002544",  # how would you describe a ball
            "massive_intent-test-002270",  # tell me how are the results of assembly elections in delhi going to come out
        ],
    ),
    (
        "Factual 'what is X' single fact",
        "Narrow what-is questions about a single fact. Surface cues: 'what is "
        "the deepest point on earth', 'what is the population of russia', "
        "'what color is a dragon fruit', 'what color are chairs', 'what sound "
        "does a dog make', 'is this the least or most important moment'.",
        "Single-fact wh-questions; distinct from tell-me-about by narrower "
        "answer space.",
        [
            "massive_intent-test-002476",  # what is the deepest point on earth
            "massive_intent-test-002526",  # what is the population of russia
            "massive_intent-test-002462",  # what color is a dragon fruit
            "massive_intent-test-002353",  # what color are chairs
            "massive_intent-test-001121",  # what sound does a dog make
            "massive_intent-test-002459",  # is this the least or most important moment in history ever
        ],
    ),
    (
        "Math / arithmetic question",
        "Arithmetic questions. Surface cues: 'what is X divided by Y', 'what's "
        "two plus two', 'one plus two equal', 'can you do nine plus two'.",
        "Distinct calculator speech-act.",
        [
            "massive_intent-test-002329",  # what is six divided by two
            "massive_intent-test-002287",  # what's two plus two
            "massive_intent-test-002411",  # one plus two equal
            "massive_intent-test-002453",  # can you do nine plus two
        ],
    ),
    (
        "Web search — bare query / look-up",
        "Bare-noun web searches and look-ups. Surface cues: 'search X', 'look "
        "up', bare 'cisco system', 'opinion petabit'.",
        "Speech-act: search-imperative; very telegraphic.",
        [
            "massive_intent-test-002944",  # search nandy
            "massive_intent-test-002571",  # look up
            "massive_intent-test-002057",  # cisco system
            "massive_intent-test-000995",  # opinion petabit
        ],
    ),
    # ---------------- CONTACTS --------------------------------------------- #
    (
        "Contacts — query groups / list",
        "Ask about contact groups. Surface cue: 'what groups are listed in my "
        "contacts'.",
        "Contact-list read intent.",
        [
            "massive_intent-test-001917",  # what groups are listed in my contacts
        ],
    ),
    (
        "Contacts — add new contact",
        "Add a contact. Surface cue: 'add business contacts to contact list'.",
        "Contact-list write intent.",
        [
            "massive_intent-test-001944",  # add business contacts to contact list
        ],
    ),
    # ---------------- CHITCHAT / SMALLTALK ---------------------------------- #
    (
        "Greeting / how-are-you smalltalk",
        "Open-ended social greeting. Surface cues: 'hello how is your day', "
        "'how has your day been today', 'has it been a busy day', 'what do you "
        "want to do today'.",
        "Pure smalltalk speech-act with no information-task.",
        [
            "massive_intent-test-001125",  # hello how is your day
            "massive_intent-test-001146",  # how has your day been today
            "massive_intent-test-001147",  # has it been a busy day
            "massive_intent-test-001101",  # what do you want to do today
        ],
    ),
    (
        "Assistant self-disclosure questions",
        "Wh- or polar-questions probing the assistant's identity, opinions, "
        "data use, smartness, relationships. Surface cues: 'do you have a "
        "boyfriend', 'are you smart', 'what do you think about future', 'what "
        "do you do with my data and information'.",
        "Speech-act: question targeting the assistant itself.",
        [
            "massive_intent-test-001158",  # do you have a boyfriend
            "massive_intent-test-001135",  # are you smart
            "massive_intent-test-001168",  # what do you think about future
            "massive_intent-test-001115",  # what do you do with my data and information
        ],
    ),
    (
        "Joke / fun request",
        "Ask for a joke or fun content. Surface cues: 'olly tell me a joke', "
        "\"what's a good joke\", 'I would like to hear some good funny jokes', "
        "tongue-twisters/riddles like the woodchuck.",
        "Entertainment-on-demand intent.",
        [
            "massive_intent-test-000023",  # olly tell me a joke
            "massive_intent-test-000229",  # what's a good joke
            "massive_intent-test-000728",  # i would like to hear some good funny jokes
            "massive_intent-test-002369",  # how much wood could a woodchuck chuck if a woodchuck could chuck wood
        ],
    ),
    (
        "Ambient declarative / chit-chat statements",
        "Free-form declaratives that aren't requests but stream-of-thought. "
        "Surface cues: 'it's time for junk food', 'a good first impression', "
        "'I mostly pay attention', 'just idle chit chat', 'I want it to be able "
        "to tell me statistics about things it's done for me', 'please know "
        "that today I had a meeting with george', 'birthday wishes', 'I want "
        "to thank everyone for the birthday wishes'.",
        "Speech-act: ambient declarative — no clear command structure; "
        "this catch-all preserves them rather than discarding to outliers.",
        [
            "massive_intent-test-000516",  # it's time for junk food
            "massive_intent-test-001276",  # a good first impression
            "massive_intent-test-000233",  # i mostly pay attention
            "massive_intent-test-002294",  # just idle chit chat
            "massive_intent-test-002293",  # i want it to be able to tell me statistics about things it's done for me
            "massive_intent-test-001129",  # please know that today i had a meeting with george
            "massive_intent-test-002600",  # birthday wishes
            "massive_intent-test-002627",  # i want to thank everyone for the birthday wishes
        ],
    ),
    # ---------------- MISC ONE-WORD / VAGUE -------------------------------- #
    (
        "Single-word / fragment commands (ambiguous control)",
        "Bare one-token commands the assistant is expected to interpret from "
        "prior context. Surface cues: 'replace', 'internet please', 'clear "
        "data'.",
        "Distinct because the speech-act is a bare verb with no complement, "
        "highly context-dependent.",
        [
            "massive_intent-test-001866",  # replace
            "massive_intent-test-002572",  # internet please
            "massive_intent-test-000831",  # clear data
        ],
    ),
]


def main() -> int:
    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    all_ids = {item["id"] for item in sample}

    seen: dict[str, str] = {}
    duplicates: list[tuple[str, str, str]] = []
    unknown: list[tuple[str, str]] = []

    out_clusters = []
    for name, description, reasoning, ids in CLUSTERS:
        for tid in ids:
            if tid not in all_ids:
                unknown.append((name, tid))
            if tid in seen:
                duplicates.append((tid, seen[tid], name))
            else:
                seen[tid] = name
        out_clusters.append(
            {
                "name": name,
                "description": description,
                "text_ids": ids,
                "reasoning": reasoning,
            }
        )

    if duplicates:
        print("ERROR: duplicate text_ids:", duplicates, file=sys.stderr)
        return 2
    if unknown:
        print("ERROR: unknown text_ids:", unknown, file=sys.stderr)
        return 2

    unclustered = sorted(all_ids - set(seen))
    cluster_sizes = [len(c["text_ids"]) for c in out_clusters]

    proposal = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "sample_size": len(sample),
        "sample_strategy": "random",
        "style": "speech-act-lexical",
        "existing_clusters_considered": False,
        "clusters": out_clusters,
        "unclustered_ids": unclustered,
        "observations": (
            "300 random voice-assistant utterances clustered by the speech-act + "
            "content combo (e.g. polite imperative on lights vs. bare-noun fragment "
            "vs. wh-question on weather). Surface cues that drive cluster identity: "
            "'play X' splits cleanly by complement type (song/artist/playlist/"
            "podcast/audiobook/radio); calendar splits cleanly by add/remove/clear/"
            "query; email splits by compose/reply/check/show; weather splits by "
            "current/forecast/precipitation/clothing-advice. Surprises: a small but "
            "real cluster of bare-fragment commands ('replace', 'clear data', "
            "'internet please') that share a speech-act type rather than any "
            "domain; a separable 'feedback / reaction to current song' cluster "
            "(declarative affect) that piggybacks on the music player; and a "
            "cluster of ambient declaratives ('it's time for junk food', 'birthday "
            "wishes') that look smalltalk-shaped but expect the assistant to act. "
            f"Cluster count: {len(out_clusters)}; sizes range from "
            f"{min(cluster_sizes)} to {max(cluster_sizes)}; unclustered: "
            f"{len(unclustered)}."
        ),
    }

    out_dir = WORKSPACE / "proposals"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:4]
    out_path = out_dir / f"prop_{stamp}_{suffix}.json"
    out_path.write_text(json.dumps(proposal, indent=2), encoding="utf-8")

    print(f"wrote {out_path}")
    print(f"clusters={len(out_clusters)} "
          f"sizes={min(cluster_sizes)}-{max(cluster_sizes)} "
          f"unclustered={len(unclustered)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
