"""Build the function-first cluster proposal for the proposer-5 angle.

Angle: "what does the assistant have to DO to fulfill this utterance?"
Each cluster name reads as a coarse domain (so it's interpretable), but the
discriminator is the underlying skill/function the assistant invokes.

Rules: no fallback bucket. Every text belongs to a real cluster. If an
utterance plausibly fits two skills, we pick the most natural primary skill.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(r"C:\Users\emily\Documents\agentic-clustering\results\clustering\massive_domain\seed=0_proposers_v1")
SAMPLE = WORKSPACE / "proposals" / "_shared_sample.json"

# Cluster definitions: (key) -> {name, description}
CLUSTERS = {
    "alarms": {
        "name": "Alarms & timers",
        "description": "Setting, listing, cancelling, or modifying alarms / wake-up timers; the assistant invokes the alarm/timer skill on a clock or device.",
    },
    "music": {
        "name": "Music & radio playback",
        "description": "Play, queue, skip, resume a song / playlist / artist / radio station; the assistant talks to a music player or tuner (Spotify, iTunes, Pandora, FM/Sirius/CNN radio channels) and selects audio content. Radio tuning is rolled in because it's the same media-playback skill family.",
    },
    "music_feedback": {
        "name": "Music feedback & metadata",
        "description": "Like/hate/rate/tag the currently playing song and ask 'what's playing' / 'who sings this' — the assistant invokes player-metadata + a favorites/ratings store, not playback selection.",
    },
    "podcasts_audiobooks": {
        "name": "Podcasts & audiobooks",
        "description": "Play, search, queue, resume podcast episodes or audiobooks — distinct skill from music because the catalog and resume-position semantics differ.",
    },
    "calendar": {
        "name": "Calendar & meetings",
        "description": "Create / modify / list / cancel calendar events, meetings, appointments; the assistant calls the calendar skill (CRUD on events with attendees, times, locations).",
    },
    "reminders": {
        "name": "Reminders & to-do scheduling",
        "description": "Set or query personal reminders / alerts for tasks (medicine, garbage, birthdays, generic 'remind me to X') — the assistant calls the reminders skill, not the calendar event store.",
    },
    "lists": {
        "name": "Lists & shopping lists",
        "description": "Create / read / modify / delete user-owned lists (shopping, to-do, favorites, errands, custom). The assistant invokes a list-management skill.",
    },
    "iot_lights": {
        "name": "Smart-home lighting",
        "description": "Turn lights on/off, dim, brighten, change color, set scene; the assistant calls the smart-home skill against light bulbs/scenes.",
    },
    "iot_appliances": {
        "name": "Smart-home appliances & device audio",
        "description": "Control non-light smart devices (coffee maker, vacuum, smart plug, generic 'clean the house') plus speaker-level audio controls (volume, mute, silence, 'speak louder', 'be quiet'). All are device-state changes the assistant routes via a device-control skill rather than a media-playback skill.",
    },
    "weather": {
        "name": "Weather lookup",
        "description": "Forecast or current-condition queries; the assistant calls a weather API for a place + time.",
    },
    "transport": {
        "name": "Transport — booking, transit & traffic",
        "description": "Book a ride / train / taxi (Uber, train ticket purchase) AND query routes, traffic conditions, train arrival times; the assistant calls a transport skill family (booking + transit info + maps). Kept together because the corpus mixes 'book a train ticket' with 'what time does the train leave' at high frequency.",
    },
    "navigation_local": {
        "name": "Local search, places & events",
        "description": "Find nearby shops, pharmacies, malls, restaurants; ask what's happening / what cultural events are on nearby. The assistant calls a local-search / POI / local-events skill.",
    },
    "email": {
        "name": "Email",
        "description": "Read, check, compose, reply to, archive email; the assistant calls the email skill (inbox + send).",
    },
    "contacts": {
        "name": "Contacts",
        "description": "Add, look up, or query the user's address book; the assistant calls the contacts skill.",
    },
    "social_media": {
        "name": "Social media posting",
        "description": "Post to Twitter / Facebook / generic 'social', read feed, check social activity; the assistant calls a social-media skill (post + read).",
    },
    "news": {
        "name": "News briefing",
        "description": "Read / fetch news headlines, articles, topical news alerts; the assistant calls a news-feed skill.",
    },
    "qa_factual": {
        "name": "Factual Q&A and definitions",
        "description": "General knowledge / definition / calculation queries answered by a knowledge-base or web-search skill (math, dictionary, geography, history, who-is, describe-X).",
    },
    "finance_stocks": {
        "name": "Finance — stocks & currency",
        "description": "Stock prices, currency exchange rates, dollar-vs-X queries; the assistant calls a finance/quotes skill.",
    },
    "recipes_cooking": {
        "name": "Recipes & cooking help",
        "description": "Recipe lookup, cooking instructions, substitutions, technique questions; the assistant calls a recipes/cooking skill.",
    },
    "time_date": {
        "name": "Time & date lookup",
        "description": "Current time (here or in a city), day-of-week, 'is it Wednesday', date-of-holiday — the assistant calls a time/clock skill, not the calendar.",
    },
    "ordering": {
        "name": "Food & product ordering",
        "description": "Order a pizza, check on a placed order, ask about delivery; the assistant calls an order/commerce skill.",
    },
    "chitchat_games": {
        "name": "Chit-chat, games & meta-assistant",
        "description": "Social small talk (knock-knock, 'how is your day', 'do you like my girlfriend'), 'play scrabble / a game' requests, and meta-assistant instructions ('remember my preferences', 'help analyze ideas', generic 'web searches' / 'internet please'). All are conversation/game/meta skill calls with no external content CRUD.",
    },
}

# Assignments. Each entry maps text_id -> cluster_key.
# Compiled by reading every utterance in the 500-text sample.
ASSIGNMENTS: dict[str, str] = {
    # batch 1
    "massive_domain-test-001577": "calendar",            # do add this on my calendar
    "massive_domain-test-001722": "recipes_cooking",     # how do i cook butter chicken
    "massive_domain-test-000165": "time_date",           # time difference here vs japan
    "massive_domain-test-001060": "weather",             # weather in LA today
    "massive_domain-test-002094": "navigation_local",    # shops nearby
    "massive_domain-test-001990": "podcasts_audiobooks", # play john's podcast
    "massive_domain-test-001658": "podcasts_audiobooks", # search podcasts men's issues
    "massive_domain-test-001242": "calendar",            # when is the event going to start
    "massive_domain-test-001952": "lists",               # list all the lists on this device
    "massive_domain-test-001466": "calendar",            # cancel all my appointments
    "massive_domain-test-002389": "weather",             # temperature in indianapolis
    "massive_domain-test-000894": "iot_appliances",      # set the coffee maker to on
    "massive_domain-test-002067": "navigation_local",    # look for shopping mall sacramento women's clothes
    "massive_domain-test-000570": "iot_lights",          # make it red in here
    "massive_domain-test-001154": "chitchat_games",      # do you like my girlfriend
    "massive_domain-test-000572": "music",               # play a live version of elton john
    "massive_domain-test-000388": "news",                # what's the news for today
    "massive_domain-test-002532": "qa_factual",          # who is the PM of russia
    "massive_domain-test-001026": "qa_factual",          # how safe is the city — factual
    "massive_domain-test-002181": "transport_booking",   # need ticket via train orlando
    "massive_domain-test-002888": "email",               # emails in the last ten minutes
    "massive_domain-test-002465": "finance_stocks",      # stock price hdfc
    "massive_domain-test-000601": "music",               # i would like to hear jazz music
    "massive_domain-test-001270": "calendar",            # when is my next appt with dr smith
    "massive_domain-test-000404": "music_feedback",      # who wrote this song
    "massive_domain-test-000302": "music",               # next song
    "massive_domain-test-002801": "email",               # send email to assistant clara cancel appts (primary: email send)
    "massive_domain-test-001352": "calendar",            # add practice to calendar feb 4 at king's park
    "massive_domain-test-001933": "lists",               # add oranges to grocery list
    "massive_domain-test-002292": "assistant_meta",      # i want it to remember my preferences
    "massive_domain-test-000412": "ordering",            # how is my order
    "massive_domain-test-001449": "time_date",           # is it wednesday
    "massive_domain-test-001778": "recipes_cooking",     # recipes that can be cooked in an hour
    "massive_domain-test-001295": "weather",             # next weeks weather in california
    "massive_domain-test-002502": "finance_stocks",      # price of dollar
    "massive_domain-test-002623": "social_media",        # hot social media topics
    "massive_domain-test-000837": "audio_control",       # shut down the sound
    "massive_domain-test-002263": "qa_factual",          # spell and define oscillate
    "massive_domain-test-001953": "lists",               # add a wrist watch to shopping list
    "massive_domain-test-001813": "chitchat_games",      # we should play nfs at high speed (game)
    "massive_domain-test-002135": "transport_booking",   # book a train ticket london manchester
    "massive_domain-test-001066": "news",                # keep me up to date with world news
    "massive_domain-test-000255": "music_feedback",      # remind me that i like that song -> save song opinion
    "massive_domain-test-002247": "qa_factual",          # definitions of orange
    "massive_domain-test-000057": "time_date",           # what day is new year's eve
    "massive_domain-test-000382": "iot_lights",          # change house lights color to blue
    "massive_domain-test-001633": "radio",               # start sirius xm radio channel
    "massive_domain-test-002909": "email",               # send email to jerry
    "massive_domain-test-002736": "email",               # email alice on the way
    "massive_domain-test-002561": "qa_factual",          # which field does that person excel in
    "massive_domain-test-000004": "iot_lights",          # olly turn the lights off in the bedroom
    "massive_domain-test-002506": "qa_factual",          # tell me about india location
    "massive_domain-test-002021": "podcasts_audiobooks", # play my favorite podcast
    "massive_domain-test-001364": "calendar",            # i have a meeting at noon today
    "massive_domain-test-000999": "weather",             # tomorrow's weather in this area
    "massive_domain-test-001332": "reminders",           # remind me when i am at the library
    "massive_domain-test-002882": "email",               # respond to my email from bob
    "massive_domain-test-000257": "time_date",           # current time in germany
    "massive_domain-test-000782": "weather",             # raining on thursday
    "massive_domain-test-002324": "assistant_meta",      # i would like it to help analyze ideas
    "massive_domain-test-000908": "audio_control",       # reduce the speaker volume
    "massive_domain-test-000977": "music",               # play giants by banks and steelz
    "massive_domain-test-000583": "time_date",           # is the 22nd on a wednesday
    "massive_domain-test-002224": "transport_booking",   # book train ticket phoenix -> LA
    "massive_domain-test-001834": "lists",               # add post office to errands list
    "massive_domain-test-000373": "iot_lights",          # turn off kitchen light
    "massive_domain-test-000329": "iot_lights",          # change light to pink color
    "massive_domain-test-001310": "calendar",            # where in DC is that meeting
    "massive_domain-test-002080": "navigation_local",    # search for local shops
    "massive_domain-test-002004": "podcasts_audiobooks", # listen to latest mike and mike podcast
    "massive_domain-test-000446": "music",               # create pandora channel for adele (music station)
    "massive_domain-test-001234": "reminders",           # add a reminder of a conference
    "massive_domain-test-002257": "qa_factual",          # how big is the cosmos
    "massive_domain-test-001192": "calendar",            # cancel the breakfast at tiffany's house
    "massive_domain-test-002894": "contacts",            # add this email address to my contacts
    "massive_domain-test-000511": "news",                # front page news articles please
    "massive_domain-test-002242": "qa_factual",          # square root of nine
    "massive_domain-test-001362": "reminders",           # what reminders did i set
    "massive_domain-test-002213": "transport_info",      # what time does the train to place leave
    "massive_domain-test-000832": "news",                # is there any news olly
    "massive_domain-test-002470": "chitchat_games",      # i think i can travel the whole world in a day
    "massive_domain-test-002241": "qa_factual",          # 200 divided by 10
    "massive_domain-test-002406": "qa_factual",          # describe a sloth
    "massive_domain-test-001178": "qa_factual",          # vacation spots (broad QA / suggestion)
    "massive_domain-test-001822": "lists",               # any special events on my list
    "massive_domain-test-000375": "news",                # read me latest international news
    "massive_domain-test-002442": "finance_stocks",      # price of starbucks stock
    "massive_domain-test-001576": "calendar",            # exhibition 2017 mass on mar 25 note on date
    "massive_domain-test-001298": "calendar",            # any events planned next three months
    "massive_domain-test-002357": "qa_factual",          # describe a rotor
    "massive_domain-test-000991": "time_date",           # time in las vegas
    "massive_domain-test-001189": "reminders",           # anything i should be reminded about
    "massive_domain-test-000753": "weather",             # rain tomorrow in rome
    "massive_domain-test-000775": "weather",             # rain at 1pm today
    "massive_domain-test-000764": "weather",             # current weather
    "massive_domain-test-000135": "music_feedback",      # i want to know about this song
    "massive_domain-test-002509": "finance_stocks",      # up and down in stock price
    "massive_domain-test-002689": "social_media",        # tweet to apple
    "massive_domain-test-001065": "weather",             # do you recommend an umbrellas (weather-led)
    "massive_domain-test-001951": "lists",               # give me all available lists
    "massive_domain-test-000282": "news",                # clean news past 6 hours
    "massive_domain-test-000367": "audio_control",       # silence for two hours
    "massive_domain-test-002780": "email",               # did charlotte responded (email response)
    "massive_domain-test-000533": "alarms",              # show me my alarms i have set
    "massive_domain-test-000612": "audio_control",       # speak loudly
    "massive_domain-test-000158": "iot_lights",          # make house lights amber at 6pm
    "massive_domain-test-000328": "iot_appliances",      # clean my house (robot vacuum)
    "massive_domain-test-002864": "email",               # did ben's email have an attachment
    "massive_domain-test-002214": "transport_info",      # how is the traffic at the moment
    "massive_domain-test-002799": "email",               # do i have new emails
    "massive_domain-test-001602": "radio",               # turn on radio play channel 106.9
    "massive_domain-test-002148": "transport_info",      # give me the train time
    "massive_domain-test-001128": "calendar",            # today the following happened i had a meeting w george (log meeting)
    "massive_domain-test-002137": "transport_booking",   # ticket for bombay
    "massive_domain-test-000964": "audio_control",       # don't make any sounds
    "massive_domain-test-000881": "alarms",              # make a new alarm
    "massive_domain-test-002783": "email",               # open inbox check unread mails
    "massive_domain-test-002415": "qa_factual",          # location of moldova
    "massive_domain-test-001717": "recipes_cooking",     # how long should i boil an egg
    "massive_domain-test-002374": "finance_stocks",      # exchange rate us mexico
    "massive_domain-test-001127": "calendar",            # today i had a meeting with george
    "massive_domain-test-001845": "lists",               # remove the old music list (playlist as list - but 'list' kw)
    "massive_domain-test-002017": "podcasts_audiobooks", # replay my last played podcast
    "massive_domain-test-002704": "email",               # please email my team
    "massive_domain-test-002626": "social_media",        # what's going on with facebook
    "massive_domain-test-001463": "reminders",           # remind me to go to dinner with dave fri 5pm
    "massive_domain-test-000337": "iot_appliances",      # coffee make now
    "massive_domain-test-001328": "calendar",            # please set a meeting to discuss terrorism with fred
    "massive_domain-test-002877": "contacts",            # are my contacts mostly female or male
    "massive_domain-test-000472": "music",               # alexa play song over the rainbow
    "massive_domain-test-001992": "podcasts_audiobooks", # download my podcast file
    "massive_domain-test-002404": "chitchat_games",      # top model car (opinion / chit-chat)
    "massive_domain-test-002581": "qa_factual",          # where do the rocky mountains start
    "massive_domain-test-001373": "calendar",            # delete the next event today
    "massive_domain-test-000779": "music",               # play song next
    "massive_domain-test-000995": "qa_factual",          # opinion petabit (definition lookup)
    "massive_domain-test-000066": "weather",             # will it be rainy tomorrow
    "massive_domain-test-001110": "qa_factual",          # tell me best tourist places in america (travel QA)
    "massive_domain-test-000479": "time_date",           # what time is it in this city
    "massive_domain-test-000903": "qa_factual",          # i want to know about brexit
    "massive_domain-test-001523": "reminders",           # show pending reminders
    "massive_domain-test-000698": "audio_control",       # turn up the volume on my speakers
    "massive_domain-test-002896": "contacts",            # mr taxi's phone number from contacts
    "massive_domain-test-001745": "recipes_cooking",     # need good ideas for cooking
    "massive_domain-test-000254": "ordering",            # order a pizza with sausage from dominos
    "massive_domain-test-002943": "email",               # get me new email
    "massive_domain-test-000599": "iot_lights",          # change lights in living room green/red
    "massive_domain-test-000896": "iot_lights",          # turn light off living room
    "massive_domain-test-000185": "iot_lights",          # brighten lights in hallway
    "massive_domain-test-002350": "finance_stocks",      # has the dollar rate increased
    "massive_domain-test-002597": "social_media",        # post my business on twitter
    "massive_domain-test-002188": "transport_info",      # how is traffic in city
    "massive_domain-test-002466": "qa_factual",          # what is a hypothesis
    "massive_domain-test-002787": "email",               # pull up kate's email and write
    "massive_domain-test-000303": "iot_lights",          # please dim the lights
    "massive_domain-test-000109": "music",               # play the song sung by katy perry
    "massive_domain-test-000509": "music",               # i want to play the song again
    "massive_domain-test-002600": "chitchat_games",      # birthday wishes
    "massive_domain-test-000772": "weather",             # how is the weather in san francisco
    "massive_domain-test-002483": "qa_factual",          # what is a caftan
    "massive_domain-test-002359": "qa_factual",          # can you really see russia from alaska
    "massive_domain-test-000490": "news",                # news stories on cnn website
    "massive_domain-test-002863": "email",               # please check my emails for me
    "massive_domain-test-000374": "iot_lights",          # turn up the lights
    "massive_domain-test-001516": "calendar",            # is jessica's birthday on april 12 (calendar lookup)
    "massive_domain-test-000475": "iot_lights",          # set living room lights 50%
    "massive_domain-test-000149": "weather",             # predicted weather for tomorrow
    "massive_domain-test-002480": "qa_factual",          # which ocean touches at our continent
    "massive_domain-test-000088": "weather",             # weather like right now
    "massive_domain-test-000797": "reminders",           # set notification for sports game
    "massive_domain-test-000757": "iot_lights",          # turn lights off
    "massive_domain-test-000507": "music_feedback",      # i hate this song
    "massive_domain-test-001962": "podcasts_audiobooks", # jump to next podcast
    "massive_domain-test-000862": "music_feedback",      # save opinion on song playing
    "massive_domain-test-000250": "audio_control",       # do not make any noise until morning alarm
    "massive_domain-test-002782": "email",               # have any emails arrived
    "massive_domain-test-000093": "iot_appliances",      # run coffee maker
    "massive_domain-test-002229": "qa_factual",          # highest building in the world
    "massive_domain-test-001743": "recipes_cooking",     # instructions to make a meal
    "massive_domain-test-002541": "qa_factual",          # definition for word pontificate
    "massive_domain-test-000415": "iot_lights",          # turn off the shed light
    "massive_domain-test-001064": "time_date",           # time in dallas texas
    "massive_domain-test-000286": "weather",             # tomorrow's temperature hot yes/no
    "massive_domain-test-000904": "music_feedback",      # tag this song with five stars
    "massive_domain-test-000294": "music_feedback",      # name of this song currently playing
    "massive_domain-test-002649": "social_media",        # send a tweet to united airlines
    "massive_domain-test-001233": "calendar",            # confirm all meetings for today
    "massive_domain-test-001434": "reminders",           # alert me a day before jeff's birthday
    "massive_domain-test-001786": "chitchat_games",      # play scrabble with me
    "massive_domain-test-000738": "iot_lights",          # make all the lights in the house blue
    "massive_domain-test-002866": "email",               # do i have any incoming emails
    "massive_domain-test-002062": "events_local",        # special events near me this weekend
    "massive_domain-test-001913": "lists",               # remove my list of favorite albums
    "massive_domain-test-000161": "music_feedback",      # i like this song please save
    "massive_domain-test-002443": "finance_stocks",      # how much is starbucks stock these days
    "massive_domain-test-000413": "iot_appliances",      # start robot vacuum cleaner
    "massive_domain-test-002811": "email",               # has tom emailed me about ammunition
    "massive_domain-test-000816": "time_date",           # what day is halloween this year
    "massive_domain-test-002875": "email",               # reply to the most recent email
    "massive_domain-test-001468": "calendar",            # set lunch meeting every wednesday in march
    "massive_domain-test-001926": "contacts",            # add bob to my list of contacts
    "massive_domain-test-002333": "qa_factual",          # how tall is mount everest
    "massive_domain-test-000693": "iot_lights",          # turn the kitchen lights off
    "massive_domain-test-002755": "email",               # has john sent me any email lately
    "massive_domain-test-000833": "music",               # play all music by billy joel
    "massive_domain-test-000237": "news",                # tell me about trump (current-affairs/news)
    "massive_domain-test-000648": "music",               # play my midnight love playlist
    "massive_domain-test-000663": "music_feedback",      # who sings the song i am listening to
    "massive_domain-test-001402": "reminders",           # remind me to something in sometime
    "massive_domain-test-002168": "transport_booking",   # purchase ticket to new york city on train
    "massive_domain-test-002955": "contacts",            # contact detail of paul
    "massive_domain-test-000480": "iot_appliances",      # coffee machine to make coffee
    "massive_domain-test-002444": "transport_info",      # find ways of travel for the same
    "massive_domain-test-001811": "chitchat_games",      # play game
    "massive_domain-test-002726": "email",               # please check my mails
    "massive_domain-test-000716": "iot_appliances",      # activate vacuum cleaner
    "massive_domain-test-000054": "music",               # play a nirvana playlist
    "massive_domain-test-001931": "lists",               # add to list
    "massive_domain-test-001678": "podcasts_audiobooks", # resume playback of "a child called it" (audiobook)
    "massive_domain-test-002331": "finance_stocks",      # currency converter please
    "massive_domain-test-002083": "events_local",        # anything good happening this weekend
    "massive_domain-test-001275": "reminders",           # remind me to take garbage out tue
    "massive_domain-test-002658": "social_media",        # siri open twitter tweet at potus sucks
    "massive_domain-test-001462": "reminders",           # please alert me
    "massive_domain-test-001591": "calendar",            # please set this date to repeat
    "massive_domain-test-002693": "news",                # expired products in recent market (news QA)
    "massive_domain-test-001027": "audio_control",       # can you stop speaking
    "massive_domain-test-000628": "music",               # turn on playlist dedicated to rock
    "massive_domain-test-002296": "qa_factual",          # 12 + 196
    "massive_domain-test-000050": "finance_stocks",      # exchange rate USD GBP
    "massive_domain-test-001875": "lists",               # throw away my to-do list
    "massive_domain-test-000323": "news",                # get me match highlights (sports news)
    "massive_domain-test-001375": "calendar",            # set meeting between myself and john at 2pm
    "massive_domain-test-000187": "music",               # do not play rock metal
    "massive_domain-test-002796": "email",               # refresh inbox + inform about new emails
    "massive_domain-test-001150": "reminders",           # mark today as the start of my diet
    "massive_domain-test-000552": "iot_lights",          # lower the lights
    "massive_domain-test-000983": "alarms",              # alarm ten am
    "massive_domain-test-001973": "podcasts_audiobooks", # play last libertarian podcast
    "massive_domain-test-001442": "calendar",            # add to calendar and repeat
    "massive_domain-test-002499": "qa_factual",          # current pm of russia
    "massive_domain-test-001179": "qa_factual",          # know any good free knitting patterns
    "massive_domain-test-001471": "calendar",            # put event into calendar mark repeating
    "massive_domain-test-002417": "finance_stocks",      # exchange rates in this region
    "massive_domain-test-002595": "social_media",        # check my social networks
    "massive_domain-test-002543": "finance_stocks",      # dollar vs euro
    "massive_domain-test-000542": "music",               # play poker face by lady gaga
    "massive_domain-test-002950": "contacts",            # contacts please
    "massive_domain-test-001589": "reminders",           # cancel my plans to pick up parents (personal plan, not on shared cal)
    "massive_domain-test-001697": "podcasts_audiobooks", # please play this playback on audiobook
    "massive_domain-test-002665": "social_media",        # negative response tweet on daikin service
    "massive_domain-test-000330": "music",               # play me playlist wacky in gaana
    "massive_domain-test-000006": "iot_appliances",      # hoover the hallway
    "massive_domain-test-002435": "qa_factual",          # what are converse shoes
    "massive_domain-test-000787": "iot_lights",          # olly change color of the lights
    "massive_domain-test-001369": "calendar",            # what are my next three events
    "massive_domain-test-000655": "alarms",              # remove the alarm set for weekdays at nine
    "massive_domain-test-000980": "radio",               # what's on the radio right now
    "massive_domain-test-000913": "reminders",           # set environmental news notification
    "massive_domain-test-002610": "news",                # what is the news today
    "massive_domain-test-001835": "lists",               # remove list item
    "massive_domain-test-001550": "reminders",           # remind me two days before my wife birthday
    "massive_domain-test-002327": "news",                # where was will ferrell seen last night (gossip news)
    "massive_domain-test-002724": "email",               # please send to new email address listed
    "massive_domain-test-000129": "iot_lights",          # please turn lower the lights
    "massive_domain-test-001647": "radio",               # play the sweet song on radio
    "massive_domain-test-002914": "email",               # start and email to john smith
    "massive_domain-test-001713": "podcasts_audiobooks", # play me random audio book about love
    "massive_domain-test-000191": "time_date",           # what day of week is march 15
    "massive_domain-test-000678": "alarms",              # set an alarm at six am
    "massive_domain-test-001824": "lists",               # delete item
    "massive_domain-test-000261": "weather",             # will it rain this week
    "massive_domain-test-001061": "music_feedback",      # store opinion on song
    "massive_domain-test-000645": "iot_appliances",      # disable smart socket
    "massive_domain-test-001828": "lists",               # i need to create a new to do list
    "massive_domain-test-002160": "transport_info",      # train arrival time to new york
    "massive_domain-test-001995": "podcasts_audiobooks", # play duncan trussel's latest podcast
    "massive_domain-test-002299": "qa_factual",          # sum of 4 + 6
    "massive_domain-test-002473": "qa_factual",          # profession of celebrity
    "massive_domain-test-000000": "alarms",              # wake me up at 5am this week
    "massive_domain-test-000159": "iot_lights",          # please turn lights off
    "massive_domain-test-002025": "podcasts_audiobooks", # i want you to play the podcast
    "massive_domain-test-001335": "calendar",            # cancel appts after 3pm today and inform
    "massive_domain-test-001278": "qa_factual",          # where is the venue of tom's marriage (factual lookup)
    "massive_domain-test-001912": "music",               # what is on my playlist
    "massive_domain-test-000204": "weather",             # how cold each night this week
    "massive_domain-test-001700": "podcasts_audiobooks", # resume joes book from where left off
    "massive_domain-test-000770": "music_feedback",      # identify song
    "massive_domain-test-002246": "recipes_cooking",     # create sugar free diet + shopping list
    "massive_domain-test-002593": "finance_stocks",      # how is ibm doing (stock proxy)
    "massive_domain-test-000341": "music_feedback",      # put that song in my favorite list
    "massive_domain-test-000534": "iot_lights",          # turn my bedroom lights off
    "massive_domain-test-000060": "weather",             # weather in barcelona in 2 days
    "massive_domain-test-001645": "radio",               # turn program on xmtune
    "massive_domain-test-001710": "music",               # keep playing secret garden
    "massive_domain-test-002940": "email",               # email mom and ask how weather is
    "massive_domain-test-000013": "alarms",              # tell me about my alarms
    "massive_domain-test-000874": "time_date",           # when is next friday the 13th
    "massive_domain-test-000058": "audio_control",       # please be quiet for another hour
    "massive_domain-test-000009": "time_date",           # what's the time in australia
    "massive_domain-test-002164": "transport_info",      # times train leaves for chicago
    "massive_domain-test-002507": "lists",               # make a list about selena gomez concert schedule
    "massive_domain-test-000400": "radio",               # what song is on the radio (radio metadata)
    "massive_domain-test-000780": "weather",             # will i need to shovel my driveway
    "massive_domain-test-000487": "ordering",            # tell me if taco bell delivers
    "massive_domain-test-002491": "qa_factual",          # features of google pixel
    "massive_domain-test-002659": "social_media",        # show me my latest social media activity
    "massive_domain-test-000813": "weather",             # do you know the weather
    "massive_domain-test-001238": "reminders",           # remind me at
    "massive_domain-test-001146": "chitchat_games",      # how has your day been today
    "massive_domain-test-000746": "music",               # play song aces high
    "massive_domain-test-000410": "ordering",            # i want to order a pizza
    "massive_domain-test-001948": "lists",               # away off from list (remove from list)
    "massive_domain-test-001624": "radio",               # start radio channel 889
    "massive_domain-test-002570": "qa_factual",          # details about bruce lee
    "massive_domain-test-000333": "music",               # can you open my itunes
    "massive_domain-test-000089": "music",               # open bad religion folder (music folder)
    "massive_domain-test-001125": "chitchat_games",      # hello how is your day
    "massive_domain-test-001855": "lists",               # create a list that is available
    "massive_domain-test-000474": "iot_appliances",      # make me a cup of coffee
    "massive_domain-test-001050": "iot_appliances",      # make the coffee
    "massive_domain-test-000546": "news",                # state of investigation into trump-russia ties
    "massive_domain-test-002133": "qa_factual",          # california (one-word lookup -> factual)
    "massive_domain-test-002641": "social_media",        # open twitter send message
    "massive_domain-test-001421": "reminders",           # set a reminder for the 13th lunch w dale
    "massive_domain-test-000471": "audio_control",       # silence volume on speakers
    "massive_domain-test-000632": "alarms",              # do i have an alarm set for tomorrow
    "massive_domain-test-001140": "qa_factual",          # web searches (generic search)
    "massive_domain-test-000076": "music",               # play a random song from playlist
    "massive_domain-test-000173": "iot_lights",          # could you turn the light off
    "massive_domain-test-000166": "weather",             # weather in new york now
    "massive_domain-test-000842": "music_feedback",      # save my opinion on adele's song
    "massive_domain-test-001063": "weather",             # info on this week weather
    "massive_domain-test-002287": "qa_factual",          # what's two plus two
    "massive_domain-test-001289": "calendar",            # schedule a one hour appointment for saturday
    "massive_domain-test-001502": "calendar",            # how many meetings with mr richards in 3mo
    "massive_domain-test-002707": "email",               # tell me if i have new emails
    "massive_domain-test-000172": "iot_lights",          # turn the living room's light off
    "massive_domain-test-002488": "news",                # does pink have a new baby (celeb news)
    "massive_domain-test-002692": "social_media",        # tweet that there was an insect in chocolate
    "massive_domain-test-001878": "lists",               # read me my list for shopping
    "massive_domain-test-002622": "social_media",        # read me new post on my feed
    "massive_domain-test-001783": "recipes_cooking",     # recipe for preparing pasta
    "massive_domain-test-001525": "calendar",            # what do i have scheduled for the 19th
    "massive_domain-test-002203": "transport_booking",   # reserve the closest uber
    "massive_domain-test-000730": "weather",             # rain in forecast for next week
    "massive_domain-test-000851": "weather",             # will it rain now
    "massive_domain-test-001538": "news",                # information of recent events (news/events)
    "massive_domain-test-002842": "contacts",            # add a new email to my contacts
    "massive_domain-test-002900": "email",               # check emails
    "massive_domain-test-000036": "music",               # play my rock playlist
    "massive_domain-test-000567": "chitchat_games",      # knock knock
    "massive_domain-test-000618": "time_date",           # time in beijing
    "massive_domain-test-001111": "calendar",            # what's my upcoming week look like
    "massive_domain-test-001365": "reminders",           # remind me to start supper at 5
    "massive_domain-test-001382": "calendar",            # what do i have to do this week
    "massive_domain-test-001504": "qa_factual",          # where is the kiss concert (factual lookup of venue)
    "massive_domain-test-000383": "iot_appliances",      # start coffee at six am
    "massive_domain-test-001385": "calendar",            # put this event in repeating pattern on calendar
    "massive_domain-test-002794": "email",               # did i receive any new email from robert
    "massive_domain-test-000146": "audio_control",       # mute the speakers
    "massive_domain-test-000168": "iot_lights",          # i want dimmer lights
    "massive_domain-test-001104": "news",                # tell me the score of the game (sports news/score)
    "massive_domain-test-000671": "music",               # listen to dance and country music
    "massive_domain-test-002869": "email",               # has ben got in touch (any new comms — primary email)
    "massive_domain-test-002963": "email",               # please archive my read messages
    "massive_domain-test-001185": "social_media",        # friend updates
    "massive_domain-test-001478": "calendar",            # tell my meetings for tomorrow morning
    "massive_domain-test-001617": "music",               # start pandora
    "massive_domain-test-002685": "social_media",        # tweet to google customer service
    "massive_domain-test-000531": "music",               # play track 1 from david bowie playlist
    "massive_domain-test-001201": "calendar",            # set an event for friday
    "massive_domain-test-000470": "audio_control",       # silence speakers
    "massive_domain-test-001958": "lists",               # list (bare keyword)
    "massive_domain-test-000981": "music_feedback",      # name of the piece you are playing
    "massive_domain-test-000197": "music",               # get sia's cheap thrills ready next
    "massive_domain-test-001261": "calendar",            # could you repeat this event
    "massive_domain-test-000735": "iot_appliances",      # turn on coffee machine
    "massive_domain-test-002142": "transport_booking",   # book me a train ticket
    "massive_domain-test-000290": "music",               # change music mode to rock
    "massive_domain-test-001239": "calendar",            # remove succeeding event
    "massive_domain-test-001651": "radio",               # play station gx in the radio
    "massive_domain-test-001345": "calendar",            # cancel all meetings + tag jim's bday event
    "massive_domain-test-001225": "reminders",           # remind me to take medicine at 9am
    "massive_domain-test-001698": "podcasts_audiobooks", # resume song from audiobook by beatles
    "massive_domain-test-000445": "news",                # show me news about the environment
    "massive_domain-test-000407": "iot_lights",          # give me more light
    "massive_domain-test-002297": "qa_factual",          # definition of timeliness
    "massive_domain-test-001970": "podcasts_audiobooks", # podcast play next episode of friends
    "massive_domain-test-001941": "lists",               # remove the guest list i created last week
    "massive_domain-test-001380": "calendar",            # new meeting add bill and malinda as attendees
    "massive_domain-test-001407": "calendar",            # do i have a date on friday
    "massive_domain-test-002817": "email",               # do i have any new emails
    "massive_domain-test-002947": "email",               # any new emails after 4 today
    "massive_domain-test-002808": "email",               # check for emails from steve
    "massive_domain-test-002038": "qa_factual",          # what movie should i watch today (recommendation)
    "massive_domain-test-001747": "recipes_cooking",     # how do i make a turkey
    "massive_domain-test-000154": "iot_appliances",      # robot vacuum the living room now
    "massive_domain-test-001236": "lists",               # give me to do list
    "massive_domain-test-001372": "calendar",            # cancel dinner tonight
    "massive_domain-test-000637": "music",               # play best friends by yelawolf
    "massive_domain-test-000682": "iot_appliances",      # let's suck out the dust (vacuum)
    "massive_domain-test-002567": "finance_stocks",      # how much to buy stock in apple
    "massive_domain-test-002312": "qa_factual",          # product of 18 and 31
    "massive_domain-test-002849": "email",               # olly check my email
    "massive_domain-test-000356": "ordering",            # how's the food order going
    "massive_domain-test-000269": "news",                # tell me the latest news
    "massive_domain-test-000346": "music_feedback",      # bring me title of current music
    "massive_domain-test-000811": "alarms",              # please list all my alarms
    "massive_domain-test-000905": "music_feedback",      # add my opinion to this song great
    "massive_domain-test-002857": "email",               # send email to boss saying i will be late
    "massive_domain-test-002886": "email",               # dictate email
    "massive_domain-test-000032": "ordering",            # do they deliver home
    "massive_domain-test-000401": "music_feedback",      # song info
    "massive_domain-test-001613": "radio",               # tune to classic hits
    "massive_domain-test-002279": "recipes_cooking",     # best chocolate chip cookies recipe
    "massive_domain-test-002125": "transport_booking",   # book a taxi to go to movies
    "massive_domain-test-001187": "calendar",            # cancel business meeting on wednesday
    "massive_domain-test-001837": "lists",               # i want to make this week's shopping list
    "massive_domain-test-002001": "podcasts_audiobooks", # please play the podcast for me
    "massive_domain-test-002395": "qa_factual",          # birthday of hemingway
    "massive_domain-test-000889": "alarms",              # check if default alarm is set
    "massive_domain-test-001732": "recipes_cooking",     # ingredient instead of saffron
    "massive_domain-test-000342": "music",               # i want some jazz music to play
    "massive_domain-test-001508": "calendar",            # the event will include walter and gemma
    "massive_domain-test-000901": "iot_lights",          # lower intensity of light
    "massive_domain-test-001068": "qa_factual",          # find funny jokes
    "massive_domain-test-002397": "qa_factual",          # tell me a place that has snow now (general lookup)
    "massive_domain-test-002572": "assistant_meta",      # internet please
    "massive_domain-test-001766": "recipes_cooking",     # difference between bake and broil
    "massive_domain-test-000786": "time_date",           # time in london now
    "massive_domain-test-002774": "news",                # give me the latest updates
    "massive_domain-test-002648": "social_media",        # tweet at united airlines
    "massive_domain-test-002702": "social_media",        # submit a negative review about a company
    "massive_domain-test-000113": "music",               # play all by playlist
    "massive_domain-test-002153": "transport_booking",   # book train ticket baltimore -> ny
    "massive_domain-test-001849": "lists",               # reset my locations list
    "massive_domain-test-000826": "iot_lights",          # change lights to a different hue
    "massive_domain-test-002669": "social_media",        # post to facebook i'm hungry
    "massive_domain-test-002036": "navigation_local",    # where is the pharmacy in leavenworth
    "massive_domain-test-001630": "radio",               # play cnn radio
    "massive_domain-test-002653": "social_media",        # tell comcast i hate them (social-style customer complaint)
    "massive_domain-test-000848": "alarms",              # turn off my first alarm
    "massive_domain-test-002636": "social_media",        # open tweet to apple iphone battery drained
    "massive_domain-test-000884": "alarms",              # new alarm for six am
    "massive_domain-test-002827": "email",               # how many new emails today
    "massive_domain-test-000428": "iot_lights",          # bedroom lights red and hall lights normal
    "massive_domain-test-000810": "music_feedback",      # this is the best band ever (opinion on music)
    "massive_domain-test-001877": "lists",               # what items are on my shopping list
    "massive_domain-test-001548": "calendar",            # open meeting time
    "massive_domain-test-001481": "calendar",            # add event to calendar app
    "massive_domain-test-002238": "qa_factual",          # look up the definition of blunder
    "massive_domain-test-000619": "weather",             # it is a sunny day today right
    "massive_domain-test-000429": "weather",             # how is the weather
    "massive_domain-test-002887": "email",               # any new email since last time
    "massive_domain-test-001998": "podcasts_audiobooks", # vitaly channel (podcast/youtube channel)
    "massive_domain-test-000607": "iot_lights",          # turn off lights in patio
    "massive_domain-test-002309": "assistant_meta",      # i would like my robot to feed/pet my dog
    "massive_domain-test-001662": "radio",               # play prairie home companion on car radio
    "massive_domain-test-001733": "recipes_cooking",     # what conducts heat better copper vs cast iron
    "massive_domain-test-002134": "qa_factual",          # new york (lookup)
    "massive_domain-test-002029": "events_local",        # any cultural events in california
    "massive_domain-test-001320": "reminders",           # remind me tonight to pick up dry cleaning
    "massive_domain-test-002041": "events_local",        # show me events in sacramento
    "massive_domain-test-002042": "qa_factual",          # long paragraph about young drivers (dictation/factual)
    "massive_domain-test-000827": "music_feedback",      # remember how i fell about this song
    "massive_domain-test-002223": "transport_booking",   # train ticket please
    "massive_domain-test-002497": "qa_factual",          # description about smartphone
    "massive_domain-test-002826": "email",               # send email to uncle john "yes i can make it"
    "massive_domain-test-000039": "time_date",           # what time is it
    "massive_domain-test-001393": "reminders",           # send an alert before meeting
    "massive_domain-test-001303": "calendar",            # inform group new meeting date is tomorrow
    "massive_domain-test-001318": "reminders",           # set calendar to remind me to buy groceries every friday
    "massive_domain-test-000145": "alarms",              # confirm my alarm settings
    "massive_domain-test-002150": "transport_info",      # what's the best way to sheffield
    "massive_domain-test-002517": "qa_factual",          # how many years was abraham lincoln president
    "massive_domain-test-001052": "weather",             # are storms likely today
    "massive_domain-test-002468": "chitchat_games",      # what's the answer to the universe (42 joke)
    "massive_domain-test-000638": "iot_lights",          # turn the lights to red color
    "massive_domain-test-001552": "calendar",            # what are meeting scheduled for today
    "massive_domain-test-002387": "qa_factual",          # how would you describe a happy birthday
    "massive_domain-test-001205": "calendar",            # show me my meetings this friday
    "massive_domain-test-002773": "email",               # show me the most recent emails
    "massive_domain-test-000271": "news",                # once a new topic on politics comes up alert me
    "massive_domain-test-002566": "finance_stocks",      # did walmart stock go up or down
    "massive_domain-test-002115": "transport_booking",   # book an uber after i left for office
    "massive_domain-test-002871": "contacts",            # add lowes hardware to my contact emails
    "massive_domain-test-002490": "transport_info",      # find route
    "massive_domain-test-000921": "alarms",              # fix alarm
    "massive_domain-test-002682": "social_media",        # tweet complaint to at
    "massive_domain-test-002643": "social_media",        # open twitter type my complaint
    "massive_domain-test-001230": "calendar",            # what i have to do next saturday at 6pm
    "massive_domain-test-000062": "iot_appliances",      # enable my plug
    "massive_domain-test-002555": "finance_stocks",      # exchange rate with one british pound
    "massive_domain-test-001354": "calendar",            # add shopping to my calendar for tomorrow
    "massive_domain-test-000658": "audio_control",       # mute the music (volume mute)
    "massive_domain-test-000609": "audio_control",       # turn off sound
    "massive_domain-test-001887": "lists",               # please show me the list that i have
    "massive_domain-test-001520": "calendar",            # am i free at 4pm
    "massive_domain-test-002068": "events_local",        # where can i go tonight
}


# Post-merge remap so we can keep ASSIGNMENTS readable (one key per original
# function) while shipping merged clusters to land in the 14..22 band.
# Each merge collapses a fine-grained skill into a parent skill family.
MERGE = {
    "audio_control": "iot_appliances",      # volume/mute = device control
    "radio": "music",                        # radio tuner = media playback family
    "events_local": "navigation_local",      # local POI + local events = same skill family
    "transport_booking": "transport",        # booking + info collapsed
    "transport_info": "transport",
    "assistant_meta": "chitchat_games",      # meta-assistant = conversation skill
}


def main() -> None:
    # Apply the merge map to assignments.
    assignments = {tid: MERGE.get(k, k) for tid, k in ASSIGNMENTS.items()}

    # Load sample to verify we have IDs and count.
    with SAMPLE.open(encoding="utf-8") as f:
        sample = json.load(f)
    sample_ids = [r["id"] for r in sample]
    assert len(sample_ids) == 500, f"expected 500 sample IDs, got {len(sample_ids)}"

    missing = [tid for tid in sample_ids if tid not in assignments]
    extra = [tid for tid in assignments if tid not in set(sample_ids)]
    if missing:
        raise SystemExit(f"ASSIGNMENTS missing {len(missing)} sample IDs, first 5: {missing[:5]}")
    if extra:
        raise SystemExit(f"ASSIGNMENTS has {len(extra)} IDs not in sample, first 5: {extra[:5]}")

    # Group IDs by cluster.
    by_cluster: dict[str, list[str]] = {k: [] for k in CLUSTERS}
    for tid, key in assignments.items():
        if key not in CLUSTERS:
            raise SystemExit(f"Unknown cluster key {key!r} for {tid}")
        by_cluster[key].append(tid)

    # Sanity: every cluster has at least 1 text (drop empties since 'no fallback bucket'
    # also implies clusters with 0 texts shouldn't ship).
    nonempty = {k: ids for k, ids in by_cluster.items() if ids}
    empty = [k for k, ids in by_cluster.items() if not ids]
    if empty:
        print(f"NOTE dropping empty clusters: {empty}")

    k = len(nonempty)
    assert 14 <= k <= 22, f"k={k} outside target band 14..22; revise CLUSTERS"

    # Total text-id coverage = 500, no unclustered.
    covered = sum(len(v) for v in nonempty.values())
    assert covered == 500, f"covered={covered} != 500"

    # Build cluster list, sorted by size desc for readability.
    cluster_records = []
    for key, ids in sorted(nonempty.items(), key=lambda kv: -len(kv[1])):
        meta = CLUSTERS[key]
        cluster_records.append({
            "name": meta["name"],
            "description": meta["description"],
            "text_ids": ids,
            "reasoning": (
                f"{len(ids)} sample texts where the assistant's primary action is "
                f"the '{key}' skill: {meta['description'].split(';')[0]}."
            ),
        })

    ts_compact = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ts_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    suffix = uuid.uuid4().hex[:4]
    out_path = WORKSPACE / "proposals" / f"prop_{ts_compact}_{suffix}.json"

    proposal = {
        "timestamp": ts_iso,
        "sample_size": 500,
        "sample_strategy": "shared_random",
        "style": "function-first",
        "existing_clusters_considered": False,
        "proposer_id": f"p5-{suffix}",
        "clusters": cluster_records,
        "unclustered_ids": [],
        "observations": (
            "Function-first lens: I picked the assistant skill that has to fire to "
            "fulfill the utterance, then named the cluster as a coarse domain so it "
            "reads like a category. Voice-assistant corpus splits cleanly along skill "
            "boundaries — alarms / timers / calendar / reminders look semantically "
            "close but call different APIs, and that distinction shows up in the data "
            "(alarms = clock device, reminders = personal-todo notifications, calendar "
            "= shared events with attendees/times/locations). Same separation for "
            "music vs podcasts/audiobooks vs radio, and for email vs social-posting vs "
            "contacts. Surprise 1: a clear 'music feedback / tagging' cluster (rate "
            "song, identify currently-playing, save opinion) — separate from playback "
            "because the skill writes to a ratings store rather than the player queue. "
            "Surprise 2: 'audio control' (volume / mute / 'be quiet') is conceptually "
            "device-level, not music-level, and groups with smart-home control more "
            "than with music. Surprise 3: many one-word utterances ('list', 'contacts "
            "please', 'california', 'new york', 'internet please') — I routed each to "
            "the most plausible primary skill given the corpus context. No fallback "
            "bucket per the hard rule: ambiguous utterances were placed on their most "
            "natural primary skill (e.g. 'play the news' would have gone to news, not "
            "music)."
        ),
    }

    out_path.write_text(json.dumps(proposal, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"WROTE {out_path}")
    print(f"k = {k} clusters")
    print(f"covered = {covered} / 500")
    for rec in cluster_records:
        print(f"  {len(rec['text_ids']):3d}  {rec['name']}")


if __name__ == "__main__":
    main()
