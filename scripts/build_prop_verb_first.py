"""Build verb-first proposal for the seed=0_auditdiv massive_intent run.

Each of the 300 sampled text_ids is hand-assigned (by the LLM that wrote this
script) to an action-verb-anchored cluster. We then emit the proposal JSON.
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

# (cluster_name, description, [text_ids])
CLUSTERS = [
    # ===== PLAY MEDIA family (verb = "play / listen / hear") =====
    ("play music",
     "User asks the assistant to play a song, artist, genre, playlist, or live music. Action verb: play / hear / listen + music object.",
     [
        "massive_intent-test-000710",  # please play the song every time i see you
        "massive_intent-test-000435",  # play my spotify dance play list
        "massive_intent-test-000141",  # play bilando
        "massive_intent-test-000054",  # play a nirvana playlist
        "massive_intent-test-000590",  # play music now please
        "massive_intent-test-000669",  # play me last year's hits
        "massive_intent-test-001686",  # could you please play the classical music for me
        "massive_intent-test-000345",  # alexa i'dl like to hear any classical except for bach or schubert
        "massive_intent-test-000588",  # play the most popular song by given artist
        "massive_intent-test-000572",  # play a live version of elton john
        "massive_intent-test-000283",  # play only all music released between... 1990 and 2000
        "massive_intent-test-000731",  # play sleepyhead by passion pit
        "massive_intent-test-000977",  # play giants by banks and steelz
        "massive_intent-test-001680",  # play jingle bells
        "massive_intent-test-001710",  # keep playing secret garden
        "massive_intent-test-000882",  # please play me songs from the eighties
        "massive_intent-test-000671",  # i'd like to listen to hear a music of dance and country
        "massive_intent-test-000691",  # i would like to hear some rap music
        "massive_intent-test-000790",  # start the dogwalking playlist
        "massive_intent-test-000675",  # rap
        "massive_intent-test-000580",  # new pop music
        "massive_intent-test-000321",  # great song for the commute
     ]),

    ("play podcast / audiobook / radio show",
     "User asks the assistant to play a podcast episode, audiobook, radio drama, or other long-form audio. Action verb: play / start / hear + non-music audio.",
     [
        "massive_intent-test-001981",  # play my most watched podcast
        "massive_intent-test-001687",  # play pride and prejudice
        "massive_intent-test-001682",  # resume animal farm
        "massive_intent-test-001976",  # play the next episode of a podcast
        "massive_intent-test-001706",  # read some more of the daisy goodwin book for me
        "massive_intent-test-001691",  # play audiobook of planets
        "massive_intent-test-001969",  # play next episode of harry potter by j. k. rowling
        "massive_intent-test-001986",  # olly start the podcast over that i started last night
        "massive_intent-test-001688",  # olly let's listen to hunt for read october
        "massive_intent-test-002019",  # lets hear the united states of anxiety podcast
        "massive_intent-test-001634",  # start radio play
     ]),

    ("tune radio station",
     "User asks the assistant to tune to a specific FM/AM radio station or frequency. Verb: tune / go to + radio station identifier.",
     [
        "massive_intent-test-001629",  # i want you to go to hot one hundred and five on the radio
        "massive_intent-test-001616",  # make the radio turn on now
        "massive_intent-test-001641",  # olly let's hear wgrr one hundred and three point five
        "massive_intent-test-001604",  # please tune nine hundred and thirty eight f. m. radio mirchi
     ]),

    ("media playback control - skip/next",
     "User controls currently-playing media by skipping or advancing. Verb: skip / next / move forward.",
     [
        "massive_intent-test-000743",  # move to the next song in the list
        "massive_intent-test-001993",  # please skip to the next podcast episode
     ]),

    ("media playback control - stop",
     "User stops currently-playing media. Verb: stop.",
     [
        "massive_intent-test-001689",  # stop play
     ]),

    ("media playback control - replay",
     "User asks for the current media to be replayed. Verb: replay.",
     [
        "massive_intent-test-001018",  # replay the musics
     ]),

    ("media playback control - mute",
     "User mutes/silences the currently-playing audio. Verb: mute.",
     [
        "massive_intent-test-000658",  # mute the music
        "massive_intent-test-000038",  # mute volume
     ]),

    ("identify currently-playing song",
     "User asks the assistant to identify a song they hear. Verb: identify.",
     [
        "massive_intent-test-000770",  # identify song
     ]),

    # ===== SEARCH / RECOMMEND MEDIA =====
    ("search/find podcasts or videos",
     "User asks the assistant to search for or list podcasts/videos to consume. Verb: find / show / search + media content.",
     [
        "massive_intent-test-001735",  # search the internet and display the youtube videos for cooking italian
        "massive_intent-test-001983",  # find the latest disney podcast
        "massive_intent-test-001987",  # show me podcasts
     ]),

    ("ask for movie recommendation / showtimes",
     "User asks what movies are available, recommended, or showing nearby. Verb: ask / look up + movies.",
     [
        "massive_intent-test-002031",  # what movie can i watch tonight on the theater here in boston
        "massive_intent-test-002096",  # look up movies near me
        "massive_intent-test-002037",  # what movies are showing in cinema today
        "massive_intent-test-002049",  # do you have any suggestion action movies
     ]),

    # ===== WEATHER QUERIES =====
    ("ask weather - current",
     "User asks the assistant what the weather is now. Verb: ask + current weather.",
     [
        "massive_intent-test-000483",  # current weather
        "massive_intent-test-001069",  # i want to know today's weather
        "massive_intent-test-000056",  # what is the weather
        "massive_intent-test-000429",  # how is the weather
        "massive_intent-test-000699",  # what is the weather going to be like today in tucson
        "massive_intent-test-000258",  # sun is shining... will we have this weather for rest of day
     ]),

    ("ask weather - forecast",
     "User asks for a future-tense weather forecast (rain, snow, tomorrow, this week). Verb: ask + forecast.",
     [
        "massive_intent-test-000730",  # is there any rain in the forecast for the next week
        "massive_intent-test-000149",  # what will be the predicted weather for tomorrow
        "massive_intent-test-000593",  # is there snow in the forecast
        "massive_intent-test-000885",  # what is the forecast for saturday
        "massive_intent-test-000775",  # is it going to rain at one p. m. today
        "massive_intent-test-000449",  # is it raining tonight
        "massive_intent-test-000726",  # am i going to be able to mow the grass this evening
     ]),

    # ===== ALARM / TIMER =====
    ("set alarm",
     "User asks the assistant to create or turn on an alarm. Verb: set / turn on / make + alarm.",
     [
        "massive_intent-test-001535",  # make an alarm for the meeting with bob at seven today
        "massive_intent-test-000156",  # set alarm for eight am
        "massive_intent-test-000850",  # turn on an alarm for three thirty p. m. today
        "massive_intent-test-000754",  # i need to get up at ten tomorrow
     ]),

    ("remove alarm",
     "User asks the assistant to delete or cancel an alarm. Verb: remove + alarm.",
     [
        "massive_intent-test-000499",  # remove set alarm
     ]),

    ("query alarm status",
     "User asks whether an alarm is set. Verb: ask + alarm state.",
     [
        "massive_intent-test-000705",  # is my reminder alarm set for dance class
     ]),

    # ===== REMINDER family =====
    ("set reminder (one-shot / recurring)",
     "User asks the assistant to create a reminder for a future moment, event, or recurring schedule. Verb: remind / set reminder.",
     [
        "massive_intent-test-001533",  # remind me to call mom every tuesday at ten am
        "massive_intent-test-001219",  # set a reminder up next month to get my oil changed
        "massive_intent-test-001247",  # remind me to start getting ready by five p. m. please
        "massive_intent-test-001494",  # remind me to water my plants every tuesday thursday and saturday
        "massive_intent-test-001440",  # set a reminder about todays faculty meeting at four
        "massive_intent-test-001493",  # remind me to pick up mark at the airport at six p. m.
        "massive_intent-test-001285",  # remind me about my anniversary in one day advance
        "massive_intent-test-001357",  # remind me i take mom to the hairdresser's thursday at 11am
        "massive_intent-test-001273",  # remind me the meeting with allen on fifteenth march
        "massive_intent-test-001456",  # remind me of my meeting on
        "massive_intent-test-001332",  # remind me when i am at the library to get a new library card
        "massive_intent-test-001526",  # send me a reminder of my meeting with tom next friday
        "massive_intent-test-001402",  # remind me to something in sometime
        "massive_intent-test-001365",  # remind me to start supper this afternoon at five
        "massive_intent-test-001462",  # please alert me
        "massive_intent-test-001549",  # i want you to make alert in the evening about upcoming meeting
        "massive_intent-test-000183",  # get hourly notification on sports news
        "massive_intent-test-000798",  # tell me when it is five p. m.
     ]),

    ("list/query reminders",
     "User asks the assistant to list, show, or describe existing reminders. Verb: show / ask + reminder list.",
     [
        "massive_intent-test-001569",  # what information do i have on reminders
        "massive_intent-test-001523",  # show pending reminders
        "massive_intent-test-001422",  # did i leave myself any reminders
        "massive_intent-test-001482",  # meeting reminders from three to five
        "massive_intent-test-001918",  # remind me of how many lists i have
     ]),

    ("remove reminder",
     "User asks the assistant to delete a reminder. Verb: remove / delete + reminder.",
     [
        "massive_intent-test-001547",  # please remove remainder
     ]),

    # ===== CALENDAR =====
    ("add calendar event",
     "User asks the assistant to add/create/schedule a calendar event or appointment. Verb: add / schedule / put / block / meet.",
     [
        "massive_intent-test-001419",  # schedule a meeting event in my calendar
        "massive_intent-test-001353",  # please add practice on feb four at king's park at two p. m.
        "massive_intent-test-001300",  # i need a meeting to be schedule with this person
        "massive_intent-test-001570",  # can you put in lee's birthday on the twenty second of june
        "massive_intent-test-001486",  # add a meeting to my calendar for nine am with tom
        "massive_intent-test-001287",  # block one hour from ten tomorrow morning
        "massive_intent-test-001329",  # meet with joe tomorrow at three
        "massive_intent-test-001196",  # set first week of june as holiday in my calendar
        "massive_intent-test-001489",  # will you send a calendar invite out... brunch at eleven am tuesday
        "massive_intent-test-001346",  # send a meeting invite to mr. ross... block my calendar friday afternoon
     ]),

    ("list/query calendar events",
     "User asks the assistant to show, list, or describe calendar appointments and schedule. Verb: show / ask + schedule.",
     [
        "massive_intent-test-001293",  # do i have any appointments
        "massive_intent-test-001243",  # give me the schedule for today's events
        "massive_intent-test-001477",  # what does my schedule look like today
        "massive_intent-test-001458",  # tell me more about my events
        "massive_intent-test-001325",  # what is the plan today
        "massive_intent-test-001257",  # what time is my doctor appointment on march thirty first
        "massive_intent-test-001859",  # recite the schedules of the list
        "massive_intent-test-002059",  # any events this weekend in pondichery
     ]),

    ("remove calendar event",
     "User asks the assistant to delete or cancel a calendar event. Verb: delete / remove / get rid of + event.",
     [
        "massive_intent-test-001881",  # remove my dentist's appointment from today's schedule
        "massive_intent-test-001381",  # open tasks delete future events
        "massive_intent-test-001308",  # can you delete the next event
        "massive_intent-test-001461",  # remove my dinner event for monday
        "massive_intent-test-001437",  # remove my next scheduled appointment please
        "massive_intent-test-001286",  # get rid of all events with jeff
     ]),

    # ===== LIST family =====
    ("create list / add list item",
     "User asks the assistant to make a new list or add an item to a list. Verb: make / add + list.",
     [
        "massive_intent-test-001915",  # make a new shopping list
        "massive_intent-test-000824",  # i want coffee everyday
     ]),

    ("query lists",
     "User asks the assistant to show or describe existing lists. Verb: show / ask + lists.",
     [
        "massive_intent-test-001904",  # olly anything else left on the list
        "massive_intent-test-001951",  # give me all available lists
     ]),

    ("remove list item",
     "User asks the assistant to delete an item from a list. Verb: delete + item.",
     [
        "massive_intent-test-001824",  # delete item
     ]),

    # ===== EMAIL family =====
    ("send email",
     "User asks the assistant to send/compose an email. Verb: send + email.",
     [
        "massive_intent-test-002826",  # please send an email to uncle john...
        "massive_intent-test-002792",  # send email to joe at a. o. l. dot com
     ]),

    ("reply to email",
     "User asks the assistant to reply to a specific email. Verb: reply.",
     [
        "massive_intent-test-002861",  # send a reply to the last email
        "massive_intent-test-002910",  # reply this mail
        "massive_intent-test-002874",  # i would like to reply to sarah's email
        "massive_intent-test-002927",  # respond with
        "massive_intent-test-002729",  # reply thank you to john
     ]),

    ("query email - inbox/check new",
     "User asks the assistant to check for new emails or show inbox state. Verb: check / ask + inbox.",
     [
        "massive_intent-test-002870",  # do i have any new emails in my inbox today
        "massive_intent-test-002900",  # check emails
        "massive_intent-test-002835",  # do i have any new emails from my contact q.
        "massive_intent-test-002827",  # how many new emails have i received today
        "massive_intent-test-002888",  # emails in the last ten minutes
        "massive_intent-test-002946",  # are there any new emails in outlook
        "massive_intent-test-002785",  # check email about my job
        "massive_intent-test-002762",  # have i received any emails in the last ten minutes
        "massive_intent-test-002714",  # has robert emails me yet
        "massive_intent-test-002856",  # tell me if i have any unread new emails
        "massive_intent-test-002915",  # have i received any emails from jeffrey burnette
        "massive_intent-test-002725",  # check email for received from mom
     ]),

    ("query email - read/show specific",
     "User asks the assistant to display the contents or details of a particular email. Verb: show / give / ask + email content.",
     [
        "massive_intent-test-002921",  # show latest email
        "massive_intent-test-002740",  # what was the subject of the last email from mom
        "massive_intent-test-002741",  # what was the last email from work
        "massive_intent-test-002954",  # give me my latest email
     ]),

    # ===== CONTACTS family =====
    ("add contact / update contact info",
     "User asks the assistant to add or modify a contact entry. Verb: add + contact (or email-to-contact).",
     [
        "massive_intent-test-002871",  # add lowes hardware to my contact emails located in cleveland
        "massive_intent-test-002744",  # add this new email with contact
        "massive_intent-test-002841",  # add a new email for julie smith
        "massive_intent-test-002902",  # please add an email for john doe to my contacts...
        "massive_intent-test-002917",  # please place my new email address to the contact
        "massive_intent-test-001944",  # add business contacts to contact list
     ]),

    ("query contact info",
     "User asks the assistant for a contact's information (email, phone, address, groups). Verb: ask + contact field.",
     [
        "massive_intent-test-001917",  # what groups are listed in my contacts
        "massive_intent-test-002743",  # what is the email address for tessa
        "massive_intent-test-002788",  # tell me about mary s.
        "massive_intent-test-002250",  # tell me billy crytals address
        "massive_intent-test-002896",  # what is mr. taxi's phone number it is in my contacts
     ]),

    # ===== SOCIAL POSTS =====
    ("post on social media",
     "User asks the assistant to post / tweet content to a social platform. Verb: tweet / post.",
     [
        "massive_intent-test-002681",  # tweet mcdonald's down reseda has terrible service
        "massive_intent-test-002661",  # send complaint via tweet to j. crew
        "massive_intent-test-002603",  # post in twitter my visit to japan
        "massive_intent-test-002642",  # tweet that product sucks
        "massive_intent-test-002668",  # i need you to tweet a complaint
        "massive_intent-test-002654",  # tweet a complaint for the online bookstore
        "massive_intent-test-002673",  # open gallery post picture name
     ]),

    ("query social media feed/trending",
     "User asks the assistant about social media feed contents, trending topics, or engagement. Verb: ask / check + social feed.",
     [
        "massive_intent-test-000755",  # what does my facebook feed look like
        "massive_intent-test-002615",  # did anyone like my photo i just posted
        "massive_intent-test-002633",  # who is doing facebook live right now
        "massive_intent-test-002632",  # please check the trending topics on twitter
     ]),

    # ===== NEWS =====
    ("ask for news headlines",
     "User asks for news, headlines, or current-events summaries. Verb: ask / tell / give + news.",
     [
        "massive_intent-test-000501",  # please tell me news related to the stock market
        "massive_intent-test-001169",  # tell interesting news
        "massive_intent-test-000511",  # front page news articles please
        "massive_intent-test-000366",  # b. b. c. world headlines
        "massive_intent-test-001056",  # i want to hear the latest world news from today
        "massive_intent-test-000767",  # what are the trending articles on the new york times
        "massive_intent-test-000299",  # give me news on president trump
        "massive_intent-test-002104",  # what's happening around town
        "massive_intent-test-002028",  # is there anything happening on jazz scene around edinburgh
        "massive_intent-test-001367",  # what's been happening for the last two months
        "massive_intent-test-001445",  # yesterday at noontime in times square what was the protest about
        "massive_intent-test-002270",  # tell me how are results of assembly elections in delhi going
        "massive_intent-test-002259",  # tell me all about hurricane
     ]),

    # ===== TIME / DATE =====
    ("ask current time",
     "User asks the assistant for the current time. Verb: ask + time.",
     [
        "massive_intent-test-000200",  # what's the time in sydney now
        "massive_intent-test-000354",  # i need to know the time right now what is it
        "massive_intent-test-000418",  # is the time currently correct for where i am located
        "massive_intent-test-000931",  # time between us and canada
     ]),

    ("ask date / day",
     "User asks the assistant for today's date, day, or day-of-something. Verb: ask + date.",
     [
        "massive_intent-test-000320",  # what month is today
        "massive_intent-test-000822",  # what day does my birthday fall on this year june 27
     ]),

    # ===== TRANSPORT - taxi/uber =====
    ("book taxi/ride",
     "User asks the assistant to book a taxi, uber, or cab. Verb: book / call + ride.",
     [
        "massive_intent-test-002120",  # call city cab to airport
        "massive_intent-test-002113",  # can you book an uber for me
        "massive_intent-test-002176",  # book me a cab going to location
        "massive_intent-test-002175",  # i need a ride home
        "massive_intent-test-002194",  # call taxi
     ]),

    # ===== TRANSPORT - train tickets =====
    ("book train/transit ticket",
     "User asks the assistant to book or purchase a train (or other transit) ticket. Verb: book / purchase + ticket.",
     [
        "massive_intent-test-002153",  # book a train ticket from baltimore maryland to new york new york
        "massive_intent-test-002135",  # please book a train ticket from london to manchester
        "massive_intent-test-002168",  # purchase ticket to new york city on train
        "massive_intent-test-002123",  # please purchase a train ticket for this weekend
     ]),

    # ===== TRANSPORT - schedule/info =====
    ("query train/transit schedule",
     "User asks the assistant about train schedules, fares, or transit info (not a booking action). Verb: ask + transit info.",
     [
        "massive_intent-test-002178",  # when does the f. train run today
        "massive_intent-test-002184",  # how much is a round trip train ticket to go to new york
        "massive_intent-test-002139",  # show me the train schedules to the metropolitan opera house
     ]),

    # ===== NAVIGATION / DIRECTIONS =====
    ("ask directions / navigation",
     "User asks the assistant for directions, fastest route, or distance to a place. Verb: ask + directions / route.",
     [
        "massive_intent-test-002193",  # navigation search
        "massive_intent-test-002441",  # how far is canada from my current location
        "massive_intent-test-002171",  # what is the fastest way to get to starbucks
        "massive_intent-test-002150",  # what's the best way to sheffield
        "massive_intent-test-002228",  # olly i need directions to madison street
        "massive_intent-test-000565",  # where is the car
        "massive_intent-test-000339",  # olly where is my food
     ]),

    # ===== LOCAL SEARCH (places near me) =====
    ("find nearby place / venue",
     "User asks the assistant to find nearby restaurants, bars, supermarkets, or other venues. Verb: ask / find + place near me.",
     [
        "massive_intent-test-002091",  # what are some restaurants near me
        "massive_intent-test-002289",  # i want to know what supermarket near me has best price on gluten free bread
        "massive_intent-test-002061",  # where is a good wine bar near me
        "massive_intent-test-001178",  # vacation spots
     ]),

    ("query restaurant ordering options",
     "User asks whether a place delivers / does takeout. Verb: ask + service capability.",
     [
        "massive_intent-test-000928",  # hey do you have home delivery
        "massive_intent-test-000232",  # does pizza hut have delivery
        "massive_intent-test-000505",  # does louie's do take out
     ]),

    # ===== ORDER FOOD =====
    ("order food",
     "User asks the assistant to order food for delivery. Verb: order + food.",
     [
        "massive_intent-test-000955",  # order pizza for delivery
        "massive_intent-test-000018",  # could you order sushi for tonight dinner
     ]),

    # ===== COOKING / RECIPES =====
    ("ask for recipe / cooking instructions",
     "User asks the assistant for a recipe or cooking instructions. Verb: ask + recipe.",
     [
        "massive_intent-test-001750",  # what's the recipe for fried chicken
        "massive_intent-test-002279",  # what is the best chocolate chip cookies recipe
        "massive_intent-test-001743",  # instructions to make a meal
        "massive_intent-test-001764",  # i need a recipe of veg pulav
     ]),

    ("operate coffee/cooking appliance",
     "User asks the assistant to run a coffee maker or cooking appliance to prepare food/drink. Verb: start / make / prepare + appliance/food.",
     [
        "massive_intent-test-001736",  # cook me some oats
        "massive_intent-test-000774",  # start the coffee machine at three
        "massive_intent-test-000792",  # can you make some coffee
        "massive_intent-test-000744",  # prepare coffee now
     ]),

    # ===== IOT - LIGHTS =====
    ("turn lights off",
     "User asks the assistant to turn off lights. Verb: turn off + lights.",
     [
        "massive_intent-test-000912",  # siri please turn the lights off in the bathroom
        "massive_intent-test-000757",  # turn lights off
        "massive_intent-test-001055",  # turn off bedroom light at nine thirty p. m.
     ]),

    ("turn lights on / increase brightness",
     "User asks the assistant to turn lights on, raise brightness, or 'turn up' lights. Verb: turn on / raise / brighten + lights.",
     [
        "massive_intent-test-000227",  # please raise the light i am not comfortable
        "massive_intent-test-000186",  # please turn up the screen brightness all the way
        "massive_intent-test-000332",  # can you turn up the lights
        "massive_intent-test-000504",  # retrieve the light for me
     ]),

    ("dim lights",
     "User asks the assistant to dim / lower light intensity. Verb: dim / lower + lights.",
     [
        "massive_intent-test-000176",  # dim the lights in the living room
        "massive_intent-test-000901",  # lower the intensity of light
     ]),

    ("change light color",
     "User asks the assistant to change the colour or hue of smart lights. Verb: change / make + light colour.",
     [
        "massive_intent-test-000210",  # olly change the lighting to blue
        "massive_intent-test-000738",  # house can you make all the lights in the house blue
        "massive_intent-test-000427",  # make the room light blue
        "massive_intent-test-000826",  # change the lights to a different hue
     ]),

    ("query light state",
     "User asks the assistant whether a light is on/off. Verb: ask + light state.",
     [
        "massive_intent-test-001047",  # did i leave the light on in the garage
     ]),

    # ===== IOT - other devices =====
    ("turn on appliance/device",
     "User asks the assistant to switch on a smart appliance (TV, vacuum, plug, radio). Verb: turn on / start / switch on + device.",
     [
        "massive_intent-test-002328",  # i'd like you to turn turn on the t. v.
        "massive_intent-test-000495",  # start the vacuum
        "massive_intent-test-000085",  # power up the plug socket one
        "massive_intent-test-000733",  # switch on the roomba
     ]),

    ("turn off appliance/device",
     "User asks the assistant to switch off a smart device (not a light). Verb: turn off / power off + device.",
     [
        "massive_intent-test-001016",  # power off please
        "massive_intent-test-000434",  # turn off my wemo socket
     ]),

    ("adjust climate / cooling",
     "User asks the assistant to set or change heating/cooling. Verb: set / lower + climate.",
     [
        "massive_intent-test-000933",  # low cooling condition
     ]),

    # ===== ASSISTANT META / SETTINGS =====
    ("change assistant voice volume",
     "User tells the assistant to speak louder or softer. Verb: speak + volume modifier.",
     [
        "massive_intent-test-000612",  # speak loudly
        "massive_intent-test-000589",  # speak softer please
     ]),

    ("tell assistant to be quiet",
     "User asks the assistant to stop speaking / wait. Verb: don't / wait + speak.",
     [
        "massive_intent-test-000134",  # don't talk until i ask you to
     ]),

    ("set assistant preferences",
     "User tells the assistant to remember a preference (likes/dislikes) for future use. Verb: note / remember + preference.",
     [
        "massive_intent-test-000541",  # please note i like jazz and hate disco
        "massive_intent-test-002292",  # i want it to remember my preferences and recommend
     ]),

    ("query assistant capabilities / self",
     "User asks the assistant about itself - feelings, age, capabilities, whether it can do X. Verb: ask + assistant-self.",
     [
        "massive_intent-test-001167",  # what is your age
        "massive_intent-test-001100",  # how are you feeling today
        "massive_intent-test-001186",  # are you real
        "massive_intent-test-001158",  # do you have a boyfriend
        "massive_intent-test-000098",  # what mistakes do you usually make
        "massive_intent-test-002236",  # can you track my emotions based on the way i speak
        "massive_intent-test-001083",  # please see what you wrote for this question
     ]),

    # ===== GENERAL KNOWLEDGE Q&A =====
    ("ask general factual question",
     "User asks the assistant a general factual / encyclopedic question (geography, history, science, definitions). Verb: ask + fact.",
     [
        "massive_intent-test-002436",  # what kinds of sunglasses are considered aviators
        "massive_intent-test-002255",  # what is the capital of nigeria
        "massive_intent-test-002457",  # explain the geographical location of india
        "massive_intent-test-002474",  # definition of velocity
        "massive_intent-test-002475",  # how big is the empire state building
        "massive_intent-test-002476",  # what is the deepest point on earth
        "massive_intent-test-002415",  # what is the location of moldova
        "massive_intent-test-002356",  # what is an electronic emissions system
        "massive_intent-test-002558",  # meaning of
        "massive_intent-test-002359",  # can you really see russia from alaska
        "massive_intent-test-002336",  # tell me where steve jobs was born
        "massive_intent-test-002390",  # can you tell me about wayne gretszky
        "massive_intent-test-002248",  # why isn't adam sandler funny
     ]),

    ("ask about person/celebrity",
     "User asks about a celebrity or public figure's activities, biography, or appearance. Verb: ask + celebrity.",
     [
        "massive_intent-test-002327",  # where was will ferrell seen last night
        "massive_intent-test-002576",  # what is denzel washington's next movie
        "massive_intent-test-002326",  # what was the last movie will smith was in
        "massive_intent-test-002419",  # what was elvis presley's birthday
     ]),

    # ===== CURRENCY / MATH / STOCKS =====
    ("ask currency exchange rate",
     "User asks for an exchange rate or currency conversion. Verb: ask + exchange rate.",
     [
        "massive_intent-test-002413",  # what is the us dollar and euro exchange rate
        "massive_intent-test-002527",  # what is the exchange rate of euro and dollar
        "massive_intent-test-002362",  # what's the exchange rates between u. s. a. and china
        "massive_intent-test-002580",  # how much is one dollar in pounds
        "massive_intent-test-002421",  # one dolla equals how much inr
     ]),

    ("ask stock price / market",
     "User asks about a stock, market, or financial figure. Verb: ask + stock.",
     [
        "massive_intent-test-002377",  # what happened to the dow jones today
        "massive_intent-test-002493",  # open stock price for name
        "massive_intent-test-002562",  # were the stocks rising or declining
     ]),

    ("do arithmetic / calculation",
     "User asks the assistant to perform a math calculation. Verb: calculate / compute.",
     [
        "massive_intent-test-002241",  # what is two hundred divided by ten
        "massive_intent-test-002307",  # what is four plus five
        "massive_intent-test-002264",  # calculate two multiplied by two
        "massive_intent-test-002288",  # alexa i've got dollars for the month how much can i spend
     ]),

    # ===== ENTERTAINMENT / CHITCHAT =====
    ("ask for joke / entertainment",
     "User asks the assistant for a joke or amusing content. Verb: tell / send + joke.",
     [
        "massive_intent-test-000023",  # olly tell me a joke
        "massive_intent-test-000022",  # send a giggle my way
        "massive_intent-test-001145",  # ask trivia questions
     ]),

    ("ask game/app to start",
     "User asks the assistant to launch / start a game or app. Verb: begin / start + app.",
     [
        "massive_intent-test-001791",  # please begin clash of clans
     ]),

    ("query order / purchase status",
     "User asks the assistant for details about an order, purchase, or transaction record. Verb: ask + order details.",
     [
        "massive_intent-test-001940",  # give me the details on purchase order
     ]),

    # ===== STATEMENT / FEEDBACK =====
    ("user makes statement / feedback",
     "User makes a self-statement, gives feedback, or notifies assistant of a fact. Not an action request. Verb: state.",
     [
        "massive_intent-test-001152",  # i got promoted today it feels so good
        "massive_intent-test-001106",  # i bought pre season baseball tickets
        "massive_intent-test-001782",  # it must be taste
        "massive_intent-test-000188",  # user friendly
        "massive_intent-test-000311",  # title
        "massive_intent-test-001139",  # what i like today
        "massive_intent-test-000962",  # the song in background is cool
        "massive_intent-test-000296",  # you like the song
        "massive_intent-test-002637",  # can you let amazon know that this new phone case is junk
        "massive_intent-test-002042",  # both young drivers and provisional drivers crash more often...
    ]),
]

# ----- assemble -----
ws = Path("C:/Users/emily/Documents/agentic-clustering/results/clustering/massive_intent/seed=0_auditdiv")
sample_path = Path("C:/Users/emily/.claude/projects/C--Users-emily-Documents-agentic-clustering/44683e58-1b15-45ad-86ed-ba9238166b1a/tool-results/bp2jsy5cz.txt")
all_ids = {r["id"] for r in json.loads(sample_path.read_text())}

assigned = set()
clusters_out = []
for name, desc, ids in CLUSTERS:
    # dedupe within
    seen = []
    for tid in ids:
        if tid in assigned:
            raise SystemExit(f"DOUBLE-ASSIGN: {tid} (cluster '{name}')")
        if tid not in all_ids:
            raise SystemExit(f"ID NOT IN SAMPLE: {tid} (cluster '{name}')")
        assigned.add(tid)
        seen.append(tid)
    clusters_out.append({
        "name": name,
        "description": desc,
        "text_ids": seen,
        "reasoning": f"Verb-first grouping: anchored on the action '{name.split()[0]}'. Texts share the same downstream behaviour (same backend call shape) despite different objects/targets.",
    })

unassigned = sorted(all_ids - assigned)
print(f"Clusters: {len(clusters_out)}")
print(f"Assigned: {len(assigned)} / {len(all_ids)}")
print(f"Unassigned: {len(unassigned)}")
for u in unassigned:
    print(f"  {u}")

# write
ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
uid = uuid.uuid4().hex[:4]
out_path = ws / "proposals" / f"prop_{ts}_{uid}.json"
out = {
    "timestamp": ts,
    "sample_size": 300,
    "sample_strategy": "random",
    "style": "verb-first",
    "existing_clusters_considered": False,
    "clusters": clusters_out,
    "unclustered_ids": unassigned,
    "observations": (
        f"Verb-first / action-centric taxonomy at {len(clusters_out)} clusters. "
        "Each cluster anchored on the action verb (play / set / send / ask / turn-on / book / etc.). "
        "Where the same verb has meaningfully different downstream behaviour (e.g. play-music vs play-podcast; "
        "set-alarm vs set-reminder vs add-calendar-event; turn-on lights vs turn-on appliance), they remain split "
        "to honour the 'fine-grained intents' directive. Where two surface verbs collapse to one backend call "
        "(book vs call vs order-a-ride for taxis; ask for news vs ask for headlines), they are merged. "
        "QUERY-vs-ACTION distinction is kept: e.g. 'add calendar event' vs 'list calendar events' vs 'remove calendar event' "
        "are three clusters in the same domain because each is a distinct verb dispatch."
    ),
}
out_path.write_text(json.dumps(out, indent=2))
print(f"\nWrote: {out_path}")
