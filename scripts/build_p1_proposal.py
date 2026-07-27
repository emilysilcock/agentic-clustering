"""
Proposer #1 (app-routing) for massive_domain seed=0_proposers_v2.

Reads the shared 300-text sample and writes an app-routing-style proposal
to proposals/prop_{timestamp}_p1ab.json. Every text is force-fit to exactly
one cluster (no orphan / other / misc). Cluster names are framed as the
DEVICE FEATURE or APP an intent router would dispatch to.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(
    r"C:/Users/emily/Documents/agentic-clustering/results/clustering/massive_domain/seed=0_proposers_v2"
)
SHARED = WORKSPACE / "_shared_sample.json"
OUT_DIR = WORKSPACE / "proposals"

# Cluster definitions: list of (name, description, reasoning, [text_ids])
# Manually curated assignment of all 300 ids based on the app/feature
# the utterance is routed to.
CLUSTERS: list[dict] = [
    {
        "name": "music-player",
        "description": (
            "Music streaming / player commands: play songs, playlists, albums, "
            "artists, like/save/tag a song, identify the now-playing track, "
            "skip / replay / remove from playlist."
        ),
        "reasoning": (
            "Routes to the music player app (Spotify, Pandora, Gaana, "
            "iTunes-style library). Distinct from radio (broadcast tuners) "
            "and from podcasts/audiobooks (separate listening apps)."
        ),
        "text_ids": [
            "massive_domain-test-001990",  # play me john's podcast -- WAIT: podcast goes to podcast cluster
        ],
    },
]

# --- Instead of inlining huge lists in a dict literal, I'll build the
# assignment dict text_id -> cluster_name and then materialize the clusters.

ASSIGN: dict[str, str] = {}


def a(cluster: str, *ids: str) -> None:
    for tid in ids:
        if tid in ASSIGN:
            raise SystemExit(f"DUPLICATE id {tid} -> {cluster} vs {ASSIGN[tid]}")
        ASSIGN[tid] = cluster


# Reset the seed cluster def -- we'll rebuild after assignments.
CLUSTERS = []

# ---- music-player (songs, playlists, song metadata, like/save) ----------
a(
    "music-player",
    "massive_domain-test-000572",  # play a live version of elton john
    "massive_domain-test-000601",  # i would like to hear jazz music
    "massive_domain-test-000404",  # who wrote this song
    "massive_domain-test-000302",  # next song
    "massive_domain-test-000255",  # remind me that i like that song (song metadata save)
    "massive_domain-test-000977",  # play giants by banks and steelz
    "massive_domain-test-000135",  # i want to know about this song
    "massive_domain-test-000446",  # could you please create a pandora channel for adele
    "massive_domain-test-000472",  # alexa play song over the rainbow
    "massive_domain-test-000779",  # play song next
    "massive_domain-test-000995",  # opinion petabit  (looks like 'opinion <song>' - song-rating phrasing)
    "massive_domain-test-000648",  # play my midnight love playlist
    "massive_domain-test-000663",  # who sings the song that i am listening to right now
    "massive_domain-test-000054",  # play a nirvana playlist
    "massive_domain-test-000628",  # turn on the playlist i have dedicated to rock music
    "massive_domain-test-000187",  # do not play rock metal
    "massive_domain-test-000542",  # play poker face by lady gaga
    "massive_domain-test-000109",  # play the song sung by katy perry
    "massive_domain-test-000509",  # i want to play the song again
    "massive_domain-test-000507",  # i hate this song
    "massive_domain-test-000862",  # save opinion on song
    "massive_domain-test-000904",  # tag this song with five stars
    "massive_domain-test-000294",  # tell me the name of this song currently playing
    "massive_domain-test-000161",  # i like this song please save
    "massive_domain-test-000833",  # play all music by billy joel
    "massive_domain-test-000770",  # identify song
    "massive_domain-test-001061",  # store opinion on song
    "massive_domain-test-000341",  # put that song in my favorite list
    "massive_domain-test-001845",  # remove the old music list
    "massive_domain-test-001913",  # please remove my list of favorite albums
    "massive_domain-test-001912",  # what is on my playlist
    "massive_domain-test-000330",  # hi google play me playlist wacky in my gaana application
)

# ---- radio-and-streaming-audio (radio channels / sirius xm / xmtune) ----
a(
    "radio-tuner",
    "massive_domain-test-001633",  # start sirius xm radio channel
    "massive_domain-test-001602",  # turn on the radio and play channel 106.9
    "massive_domain-test-000980",  # what's on the radio right now
    "massive_domain-test-001647",  # play the sweet song on radio
    "massive_domain-test-001645",  # turn program on xmtune
)

# ---- podcast-and-audiobook-player (spoken-audio app) -------------------
a(
    "podcast-and-audiobook-player",
    "massive_domain-test-001990",  # play me john's podcast
    "massive_domain-test-001658",  # search for podcasts that cover men's issues
    "massive_domain-test-002021",  # hey play my favorite podcast from list
    "massive_domain-test-002004",  # latest mike and mike podcast
    "massive_domain-test-002017",  # replay my last played podcast
    "massive_domain-test-001962",  # jump to next podcast
    "massive_domain-test-001973",  # play last libertarian podcast
    "massive_domain-test-001992",  # download podcast file and play
    "massive_domain-test-001995",  # play duncan trussel's latest podcast
    "massive_domain-test-002025",  # i want you to play the podcast
)

# ---- audiobook-player (merged into podcast-player as spoken-audio-player) ----
a(
    "podcast-and-audiobook-player",
    "massive_domain-test-001678",  # resume playback of 'a child called it'
    "massive_domain-test-001700",  # resume joe's book where i left off
    "massive_domain-test-001697",  # play this playback on audiobook
    "massive_domain-test-001713",  # play a random audio book about love
)

# ---- weather-app -------------------------------------------------------
a(
    "weather-app",
    "massive_domain-test-001060",  # weather today in LA
    "massive_domain-test-002389",  # temperature in indianapolis
    "massive_domain-test-001295",  # next week's weather in california
    "massive_domain-test-000999",  # tomorrow's weather in this area
    "massive_domain-test-000782",  # will it be raining on thursday
    "massive_domain-test-000753",  # will there be rain tomorrow in rome
    "massive_domain-test-000775",  # rain at one pm today
    "massive_domain-test-000764",  # current weather
    "massive_domain-test-001065",  # do you recommend an umbrellas (weather-driven)
    "massive_domain-test-000066",  # will it be rainy tomorrow
    "massive_domain-test-000286",  # tomorrow's temperature is to be hot yes or no
    "massive_domain-test-000772",  # weather in san francisco
    "massive_domain-test-000149",  # predicted weather for tomorrow
    "massive_domain-test-000088",  # what is the weather like right now
    "massive_domain-test-000204",  # how cold will it get each night this week
    "massive_domain-test-000060",  # tell me the weather in barcelona
    "massive_domain-test-000261",  # will it rain this week
)

# ---- smart-home-lights -------------------------------------------------
a(
    "smart-home-lights",
    "massive_domain-test-000570",  # make it red in here
    "massive_domain-test-000382",  # change my house lights color to blue
    "massive_domain-test-000004",  # olly turn the lights off in the bedroom
    "massive_domain-test-000373",  # turn off kitchen light
    "massive_domain-test-000329",  # change the light to pink color
    "massive_domain-test-000158",  # make house lights amber at 6pm
    "massive_domain-test-000599",  # change living room lights to green and red
    "massive_domain-test-000896",  # turn the light off in the living room
    "massive_domain-test-000185",  # brighten the lights in the hallway
    "massive_domain-test-000303",  # please dim the lights
    "massive_domain-test-000374",  # i can't see turn up the lights
    "massive_domain-test-000475",  # set living room lights to 50%
    "massive_domain-test-000757",  # turn lights off
    "massive_domain-test-000738",  # make all lights in the house blue
    "massive_domain-test-000693",  # turn the kitchen lights off
    "massive_domain-test-000415",  # turn off the shed light
    "massive_domain-test-000787",  # change the color of the lights
    "massive_domain-test-000552",  # lower the lights please
    "massive_domain-test-000129",  # turn lower the lights
    "massive_domain-test-000159",  # please turn lights off
    "massive_domain-test-000534",  # turn my bedroom lights off
)

# ---- smart-home-appliances (coffee maker, vacuum, sockets, "clean house") ----
a(
    "smart-home-appliances",
    "massive_domain-test-000894",  # set the coffee maker to on
    "massive_domain-test-000337",  # coffee make now
    "massive_domain-test-000093",  # run coffee maker
    "massive_domain-test-000480",  # please get the coffee machine to make me some coffee
    "massive_domain-test-000413",  # start robot vacuum cleaner
    "massive_domain-test-000716",  # activate the vacuum cleaner
    "massive_domain-test-000006",  # hoover the hallway
    "massive_domain-test-000328",  # clean my house
    "massive_domain-test-000645",  # disable smart socket
)

# ---- volume-and-audio-control (device speaker volume / mute / TTS) -----
a(
    "device-audio-control",
    "massive_domain-test-000837",  # shut down the sound
    "massive_domain-test-000908",  # please reduce the speaker volume
    "massive_domain-test-000367",  # silence for two hours
    "massive_domain-test-000612",  # speak loudly
    "massive_domain-test-000964",  # don't make any sounds
    "massive_domain-test-000698",  # turn up the volume on my speakers
    "massive_domain-test-001027",  # can you please stop speaking
    "massive_domain-test-000250",  # do not make any noise until morning alarm
)

# ---- alarms-and-wakeups ------------------------------------------------
a(
    "alarms",
    "massive_domain-test-000533",  # show me my alarms
    "massive_domain-test-000881",  # make a new alarm
    "massive_domain-test-000983",  # alarm ten am
    "massive_domain-test-000678",  # set an alarm at six am
    "massive_domain-test-000000",  # wake me up at five am this week
    "massive_domain-test-000655",  # remove the alarm set for weekdays at nine
)

# ---- calendar-and-meetings ---------------------------------------------
a(
    "calendar-events",
    "massive_domain-test-001577",  # do add this on my calendar
    "massive_domain-test-001242",  # when is the event going to start
    "massive_domain-test-001466",  # cancel all my appointments
    "massive_domain-test-001352",  # add practice to calendar feb 4
    "massive_domain-test-001270",  # next appointment with doctor smith
    "massive_domain-test-001364",  # i have a meeting at noon today
    "massive_domain-test-001310",  # where is meeting in d.c. at 1pm friday
    "massive_domain-test-001234",  # add a reminder of a conference for tomorrow ny
    "massive_domain-test-001192",  # cancel the breakfast at tiffany's house
    "massive_domain-test-001576",  # exhibition 2017 make a note on the date
    "massive_domain-test-001298",  # events planned for next three months
    "massive_domain-test-001822",  # any special events on my list
    "massive_domain-test-001373",  # delete the next event today
    "massive_domain-test-001328",  # set a meeting to discuss terrorism with fred
    "massive_domain-test-001463",  # remind me to go to dinner with dave friday 5pm
    "massive_domain-test-001233",  # confirm all meetings for today
    "massive_domain-test-001468",  # set lunch meeting at 12pm every wednesday march
    "massive_domain-test-001375",  # set a meeting between myself and john at 2pm tomorrow
    "massive_domain-test-001442",  # add to calendar and repeat
    "massive_domain-test-001471",  # put event into my calendar mark as repeating
    "massive_domain-test-001591",  # please set this date to repeat
    "massive_domain-test-001369",  # what are my next three events
    "massive_domain-test-001589",  # cancel my plans to pick up my parents from the airport
    "massive_domain-test-001335",  # cancel all my appointments after 3pm today
    "massive_domain-test-001128",  # today the following happened to me, meeting with george
    "massive_domain-test-001127",  # today i had a meeting with george
    "massive_domain-test-001278",  # where is the venue marriage of tom
)

# ---- reminders ---------------------------------------------------------
a(
    "reminders",
    "massive_domain-test-001362",  # what reminders did i set
    "massive_domain-test-001189",  # is there anything i should be reminded about
    "massive_domain-test-001332",  # remind me at library to get card
    "massive_domain-test-001434",  # alert me day before jeff's birthday
    "massive_domain-test-001523",  # show pending reminders
    "massive_domain-test-001462",  # please alert me
    "massive_domain-test-001275",  # remind me to take the garbage out on tuesday
    "massive_domain-test-001402",  # remind me to something in sometime
    "massive_domain-test-001550",  # remind me two days before my wife birthday
    "massive_domain-test-001150",  # mark today as the start of my diet
)

# ---- lists-and-todos ---------------------------------------------------
a(
    "lists-and-todos",
    "massive_domain-test-001952",  # list all the lists on this device
    "massive_domain-test-001933",  # add oranges to my grocery list
    "massive_domain-test-001953",  # add wrist watch to the shopping list
    "massive_domain-test-001834",  # add post office to errands for saturday
    "massive_domain-test-001951",  # give me all available lists
    "massive_domain-test-001875",  # throw away my to do list
    "massive_domain-test-001828",  # create a new to do list
    "massive_domain-test-001931",  # add to list
    "massive_domain-test-001835",  # remove list item
    "massive_domain-test-001824",  # delete item
)

# ---- email-app ---------------------------------------------------------
a(
    "email-app",
    "massive_domain-test-002888",  # emails in the last ten minutes
    "massive_domain-test-002801",  # send email to assistant clara to cancel apts
    "massive_domain-test-002947",  # new emails received after 4pm today
    "massive_domain-test-002909",  # send email to jerry ask time he be home
    "massive_domain-test-002736",  # email alice to let her know
    "massive_domain-test-002882",  # respond to my email from bob
    "massive_domain-test-002894",  # add email address to contacts and send email
    "massive_domain-test-002864",  # did ben's email have an attachment
    "massive_domain-test-002799",  # do i have new emails
    "massive_domain-test-002783",  # open inbox to check unread mails
    "massive_domain-test-002780",  # did charlotte respond
    "massive_domain-test-002704",  # please email my team
    "massive_domain-test-002868",  # did i get an email from paul
    "massive_domain-test-002889",  # interview time from yesterday's email
    "massive_domain-test-002857",  # send email to boss saying i will be late
    "massive_domain-test-002787",  # pull up kate's email and write reply
    "massive_domain-test-002941",  # write an email for john at gmail dot com
    "massive_domain-test-002782",  # any emails arrived in last 15 minutes
    "massive_domain-test-002755",  # has john sent me any email lately
    "massive_domain-test-002769",  # check email containing job listing
    "massive_domain-test-002726",  # please check my mails
    "massive_domain-test-002829",  # please sent this email to my friend
    "massive_domain-test-002758",  # i want to send an email to my family
    "massive_domain-test-002759",  # check gmail
    "massive_domain-test-002874",  # reply to sarah's email
    "massive_domain-test-002712",  # email chelsea
    "massive_domain-test-002904",  # what did i tell susan in my last email
    "massive_domain-test-002873",  # write a reply to my mother's email
    "massive_domain-test-002779",  # who mailed me yesterday
    "massive_domain-test-002861",  # send a reply to the last email
    "massive_domain-test-002931",  # any new email from john
    "massive_domain-test-002971",  # do i have any new email from joe
)

# ---- contacts: merged into email-app (single contacts utterance) -------
a(
    "email-app",
    "massive_domain-test-001926",  # add bob to my list of contacts
)

# ---- social-media-app --------------------------------------------------
a(
    "social-media-app",
    "massive_domain-test-002623",  # hot social media topics
    "massive_domain-test-002626",  # tell me what's going on with facebook
    "massive_domain-test-002689",  # tweet to apple about iphone repairs
    "massive_domain-test-002597",  # post my business on twitter
    "massive_domain-test-002649",  # send tweet to united airlines (anger)
    "massive_domain-test-002658",  # siri open twitter tweet at potus sucks
    "massive_domain-test-002665",  # negative response tweet on daikin service
    "massive_domain-test-002595",  # check my social networks
)

# ---- news-reader -------------------------------------------------------
a(
    "news-reader",
    "massive_domain-test-000388",  # what's the news for today
    "massive_domain-test-001066",  # keep me up to date with world news
    "massive_domain-test-000511",  # front page news articles please
    "massive_domain-test-000282",  # clean news in past six hours
    "massive_domain-test-000832",  # is there any news olly
    "massive_domain-test-000375",  # read me the latest international news
    "massive_domain-test-000490",  # news stories on cnn website
    "massive_domain-test-000903",  # i want to know about the brexit
    "massive_domain-test-000237",  # tell me about trump
    "massive_domain-test-000913",  # set environmental news notification
    "massive_domain-test-002610",  # what is the news today
    "massive_domain-test-000797",  # set a notification for sports game
    "massive_domain-test-000323",  # get me match highlights
    "massive_domain-test-002327",  # where was will ferrell seen last night (celeb news)
)

# ---- stocks-and-finance ------------------------------------------------
a(
    "stocks-and-finance",
    "massive_domain-test-002465",  # stock price of hdfc
    "massive_domain-test-002502",  # price of dollar
    "massive_domain-test-002442",  # price of starbuck's stock
    "massive_domain-test-002509",  # up/down in stock price
    "massive_domain-test-002374",  # exchange rate between us and mexico
    "massive_domain-test-002350",  # has the dollar rate increased
    "massive_domain-test-002443",  # how much is starbuck's stock these days
    "massive_domain-test-002417",  # exchange rates in this region
    "massive_domain-test-002543",  # how much is dollar worth vs euro
    "massive_domain-test-002593",  # how is i.b.m. doing
    "massive_domain-test-000050",  # exchange rate us dollar to pound
    "massive_domain-test-002331",  # currency converter please
)

# ---- transit-and-traffic (train booking, train schedules, traffic) ----
a(
    "transit-and-traffic",
    "massive_domain-test-002181",  # train ticket orlando from hollywood
    "massive_domain-test-002135",  # book train ticket london to manchester
    "massive_domain-test-002224",  # book train ticket phoenix to la
    "massive_domain-test-002213",  # what time does the train to place leave
    "massive_domain-test-002148",  # give me the train time
    "massive_domain-test-002137",  # ticket for bombay
    "massive_domain-test-002168",  # purchase ticket to new york city on train
    "massive_domain-test-002160",  # train arrival time to new york
    "massive_domain-test-002444",  # find the ways of travel for the same
)

# ---- traffic: merged into transit-and-traffic ---------------------------
a(
    "transit-and-traffic",
    "massive_domain-test-002214",  # how is the traffic at the moment
    "massive_domain-test-002188",  # how is traffic in city
)

# ---- local-search-and-events (nearby shops, events near me, vacation/tourism) ----
a(
    "local-search",
    "massive_domain-test-002094",  # what shops are nearby
    "massive_domain-test-002067",  # shopping mall in sacramento for women's clothes
    "massive_domain-test-002080",  # search for local shops
    "massive_domain-test-002062",  # special events near me this weekend
    "massive_domain-test-002083",  # anything good happening this weekend in the area
    "massive_domain-test-001178",  # vacation spots
    "massive_domain-test-001110",  # best tourist places to visit in america
    "massive_domain-test-001026",  # how safe is the city regarding law and order
)

# ---- food (recipes + ordering, single food app) -----------------------
a(
    "food-recipes-and-ordering",
    "massive_domain-test-001722",  # how do i cook butter chicken
    "massive_domain-test-001778",  # recipes that can be cooked in an hour
    "massive_domain-test-001717",  # how long should i boil an egg
    "massive_domain-test-001745",  # i need good ideas for cooking
    "massive_domain-test-001743",  # instructions to make a meal
    "massive_domain-test-002246",  # sugar free diet + shopping list (mostly recipe)
)

# ---- food ordering: merged into food-recipes-and-ordering -------------
a(
    "food-recipes-and-ordering",
    "massive_domain-test-000254",  # order a pizza with sausage from domino's
    "massive_domain-test-000412",  # how is my order
)

# ---- clock-and-world-time ----------------------------------------------
a(
    "clock-and-time",
    "massive_domain-test-000165",  # time difference between here and japan
    "massive_domain-test-000257",  # current time in germany
    "massive_domain-test-000991",  # what is the time in las vegas
    "massive_domain-test-000479",  # what time is it in this city
    "massive_domain-test-001064",  # what time is it in dallas texas
)

# ---- calendar-date-lookup (what-day-is-X) ------------------------------
a(
    "date-lookup",
    "massive_domain-test-001449",  # is it wednesday
    "massive_domain-test-000057",  # new year's eve this year
    "massive_domain-test-000583",  # is the 22nd on a wednesday
    "massive_domain-test-000816",  # what day is halloween this year
    "massive_domain-test-000191",  # what day of week does 15th of march fall on
    "massive_domain-test-001516",  # is jessica's birthday on april 12
    "massive_domain-test-002790",  # is it anyone i knows birthday this month
    "massive_domain-test-002600",  # birthday wishes
)

# ---- general-knowledge-qa (encyclopedic / facts / definitions / math) ----
a(
    "general-knowledge-qa",
    "massive_domain-test-002532",  # prime minister of russia
    "massive_domain-test-002247",  # definitions of orange
    "massive_domain-test-002263",  # spell and define oscillate
    "massive_domain-test-002506",  # tell me about india location
    "massive_domain-test-002561",  # in which field does that person excel
    "massive_domain-test-002257",  # how big is the cosmos
    "massive_domain-test-002242",  # square root of nine
    "massive_domain-test-002241",  # 200 divided by 10
    "massive_domain-test-002406",  # describe a sloth
    "massive_domain-test-002357",  # describe a rotor
    "massive_domain-test-002415",  # location of moldova
    "massive_domain-test-002470",  # i think i can travel the whole world in a day
    "massive_domain-test-002581",  # where do the rocky mountains start
    "massive_domain-test-002359",  # can you really see russia from alaska
    "massive_domain-test-002480",  # which ocean touches at our continent
    "massive_domain-test-002296",  # 12 + 196
    "massive_domain-test-002299",  # sum of four and six
    "massive_domain-test-002229",  # highest building in the world
    "massive_domain-test-002333",  # how tall is mount everest
    "massive_domain-test-002466",  # what is a hypothesis
    "massive_domain-test-002435",  # what are converse shoes
    "massive_domain-test-002483",  # what is a caftan
    "massive_domain-test-002541",  # define pontificate
    "massive_domain-test-002473",  # tell me the profession of celebrity
    "massive_domain-test-002499",  # prime minister of russia (dup-ish phrasing)
    "massive_domain-test-002693",  # expired products in recent market
    "massive_domain-test-002404",  # top model car
    "massive_domain-test-001179",  # do you know any good free knitting patterns
)

# ---- assistant-config / chit-chat / preferences ------------------------
a(
    "assistant-config",
    "massive_domain-test-001154",  # hey olly do you like my girlfriend
    "massive_domain-test-002292",  # remember my preferences and recommend
    "massive_domain-test-002324",  # i would like it to help analyze ideas
    "massive_domain-test-001811",  # play game
    "massive_domain-test-001813",  # we should play nfs at high speed
    "massive_domain-test-001786",  # play scrabble with me
)

# ---------------------------------------------------------------------
# Verify coverage
# ---------------------------------------------------------------------
sample = json.loads(SHARED.read_text(encoding="utf-8"))
all_ids = {x["id"] for x in sample}
covered = set(ASSIGN.keys())

missing = sorted(all_ids - covered)
extra = sorted(covered - all_ids)

if extra:
    raise SystemExit(f"FATAL: {len(extra)} ids assigned but not in sample: {extra[:5]}")
if missing:
    raise SystemExit(
        f"FATAL: {len(missing)} ids missing from assignment (no orphans allowed):\n"
        + "\n".join(missing[:30])
    )

# Build clusters in a stable order matching first-mention.
order: list[str] = []
seen: set[str] = set()
for tid in ASSIGN:  # insertion order = call order
    c = ASSIGN[tid]
    if c not in seen:
        order.append(c)
        seen.add(c)

DESCRIPTIONS = {
    "music-player": (
        "Music streaming / library commands: play songs/playlists/artists/genres, "
        "skip-replay-next, identify or rate the now-playing track, "
        "save/remove favorites and playlists. Routes to the music app "
        "(Spotify, Pandora, Gaana, iTunes-style library).",
        "Distinct from radio tuners, podcast and audiobook players.",
    ),
    "radio-tuner": (
        "Radio / streaming-channel tuner: turn on a radio channel, "
        "switch sirius/xm channels, ask what's currently on the radio.",
        "Routes to a broadcast-tuner app, not the on-demand music player.",
    ),
    "podcast-and-audiobook-player": (
        "Spoken-audio player: podcasts (play / replay / skip / download / "
        "search) and audiobooks (resume a specific book, play a random "
        "audiobook).",
        "Combined because podcast and audiobook players occupy the same "
        "spoken-audio slot on most devices; kept separate from music "
        "(library / song-centric) and from radio (broadcast tuner).",
    ),
    "weather-app": (
        "Weather queries: current conditions, forecast (today/tomorrow/this week), "
        "temperature in a city, will it rain, umbrella recommendation.",
        "Single device feature (weather app / forecast widget).",
    ),
    "smart-home-lights": (
        "Smart-home lighting: turn lights on/off in a room, brighten / dim, "
        "change colour, schedule a colour at a time.",
        "Routes to the smart-bulbs subsystem of the home-automation app.",
    ),
    "smart-home-appliances": (
        "Other smart-home appliances: coffee maker, robot vacuum / hoover, "
        "smart sockets, generic 'clean my house'.",
        "Same home-automation umbrella as lights but a different device handler.",
    ),
    "device-audio-control": (
        "Device-level audio controls: speaker volume up/down, mute, silence, "
        "speak louder, stop speaking.",
        "Routes to the OS audio/TTS subsystem, not to a content app like music.",
    ),
    "alarms": (
        "Wake-up alarms: set / list / remove alarms, 'wake me up at X'.",
        "Distinct app from reminders and from calendar events.",
    ),
    "calendar-events": (
        "Calendar / meetings / appointments: add events, query upcoming events, "
        "schedule meetings with people, cancel appointments, find meeting venue, "
        "log a meeting that already happened.",
        "Routes to the calendar app; different from one-off reminders and "
        "from to-do lists.",
    ),
    "reminders": (
        "Standalone reminders: set / list / query reminders and alerts "
        "(time-based or location-based), 'mark today as start of X'.",
        "Routes to the Reminders app, separate from the calendar.",
    ),
    "lists-and-todos": (
        "Lists app: to-do lists, shopping / grocery / errand lists, "
        "list all my lists, add/remove items.",
        "Routes to a Lists/Notes app; not the calendar or reminders.",
    ),
    "email-app": (
        "Email client: send / reply / forward email, check inbox, "
        "ask whether sender X mailed, query attachments, count new mails, "
        "look up info inside an email; also contacts/address-book ops "
        "(adding a person to contacts) which share the same mail app.",
        "Routes to the mail app (Gmail, generic inbox).",
    ),
    "social-media-app": (
        "Social-media app: post / tweet / reply on Twitter, "
        "check Facebook, browse social networks, social-trend queries.",
        "Routes to a social-media client (Twitter/Facebook), distinct from "
        "general news.",
    ),
    "news-reader": (
        "News / headlines reader: today's news, world / international news, "
        "news from a named source (CNN), topical news (brexit, trump), "
        "sports highlights, notifications for a news topic.",
        "Routes to a news / headline-feed app.",
    ),
    "stocks-and-finance": (
        "Stocks, currency rates, exchange rates, currency converter.",
        "Routes to a finance / markets app.",
    ),
    "transit-and-traffic": (
        "Train ticket booking, train schedule lookups, generic ways-to-travel "
        "queries, and live traffic-condition queries.",
        "Routes to a transit / maps app (booking + traffic in one slot since "
        "both are travel-time-centric and the sample is small on each).",
    ),
    "local-search": (
        "Local discovery: nearby shops, malls, weekend events near me, "
        "vacation / tourism spot suggestions, city safety.",
        "Routes to a local-search / Yelp-style app, distinct from web "
        "knowledge queries.",
    ),
    "food-recipes-and-ordering": (
        "Food: recipe lookups, cooking instructions, diet plans, plus "
        "food-delivery ordering and order-status queries.",
        "Routes to the device's food app — merged because both recipe lookup "
        "and food ordering live under the same 'food' intent on most devices "
        "and the sample volume is small.",
    ),
    "clock-and-time": (
        "World-clock queries: current time in another city / country, "
        "time-zone differences.",
        "Routes to the clock app; separate from calendar date lookups.",
    ),
    "date-lookup": (
        "Calendar-style date questions that don't add events: what day is X, "
        "is the 22nd a wednesday, is jessica's birthday on april 12, "
        "anyone's birthday this month, generic birthday-wishes.",
        "Routes to the calendar's date-lookup feature rather than the "
        "event-store; kept separate from world-clock.",
    ),
    "general-knowledge-qa": (
        "Encyclopedic / web-search Q&A: facts, definitions, spellings, math, "
        "geography, celebrity / product facts, random trivia.",
        "Routes to the general web-search / knowledge skill, the catch-all "
        "for factual questions not handled by a more specific app.",
    ),
    "assistant-config": (
        "Talking-to-the-assistant directly: chit-chat / opinion of the user, "
        "preference/personalisation settings, and games launched through "
        "the assistant ('play scrabble', 'play game', 'play nfs').",
        "Routes to the assistant's own settings / games shell rather than a "
        "content app. Forced grouping for the small chit-chat + games tail.",
    ),
}

clusters_out: list[dict] = []
for cname in order:
    desc, reasoning = DESCRIPTIONS[cname]
    ids = [tid for tid, c in ASSIGN.items() if c == cname]
    clusters_out.append(
        {
            "name": cname,
            "description": desc,
            "text_ids": ids,
            "reasoning": reasoning,
        }
    )

print(f"clusters: {len(clusters_out)}")
for c in clusters_out:
    print(f"  {c['name']:25s} {len(c['text_ids'])}")

# Sanity check k
k = len(clusters_out)
if not (14 <= k <= 22):
    raise SystemExit(f"FATAL: k={k} outside 14-22 target band")

# Build proposal payload
ts_iso = datetime.now(timezone.utc).isoformat()
ts_fname = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

payload = {
    "timestamp": ts_iso,
    "sample_size": 300,
    "sample_strategy": "shared-fixed-seed-0",
    "style": "app-routing",
    "existing_clusters_considered": False,
    "clusters": clusters_out,
    "unclustered_ids": [],
    "observations": (
        "Read each utterance as an intent-router would: 'which app on the device "
        "handles this?'. Audio splits four ways (music vs radio vs podcast vs "
        "audiobook) because they're four different player apps even though "
        "they're all 'play X'. Smart-home splits into lights vs other "
        "appliances because they're separate handlers in most home-automation "
        "stacks. Calendar (events with people / times), reminders "
        "(time- or location-based alerts), and lists/to-dos are three distinct "
        "apps despite frequent overlap in vocabulary. 'Birthday on april 12' "
        "and 'is it wednesday' go to a calendar-date-lookup feature rather "
        "than the event store, kept separate from the world-clock app. "
        "Surprises: (a) a handful of utterances that read as social chit-chat "
        "or game-launches ('hey olly do you like my girlfriend', 'play "
        "scrabble', 'play nfs at high speed') were force-fit to an "
        "assistant-config bucket since per the hard rule no orphan/other "
        "cluster is allowed; (b) the 300-sample is heavily email-skewed "
        "(~32 emails) and music-skewed (~32 music) -- a router would see "
        "those as its two hottest endpoints. unclustered_ids is empty; "
        f"all 300 ids are assigned across {k} clusters."
    ),
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / f"prop_{ts_fname}_p1ab.json"
out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"\nWROTE {out_path}")
print(f"total assigned: {sum(len(c['text_ids']) for c in clusters_out)}")
