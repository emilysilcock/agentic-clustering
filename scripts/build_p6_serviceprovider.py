"""Build proposer #6 (service-provider lens) proposal for massive_domain auditdiv run."""
import json
import uuid
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(r"C:\Users\emily\Documents\agentic-clustering\results\clustering\massive_domain\seed=0_auditdiv")
SAMPLE = WORKSPACE / "_shared_sample.json"

with open(SAMPLE, encoding="utf-8") as f:
    items = json.load(f)

assert len(items) == 300, f"expected 300, got {len(items)}"

# Service-provider lens: which backing service/skill would handle each utterance.
# Each cluster is a service. Every text must land in exactly one service cluster.

clusters = [
    {
        "name": "Music service",
        "description": "Requests handled by the music-playback backing service: play / skip / rate songs and albums, choose genres, manage playlists, query song info.",
        "text_ids": [
            "massive_domain-test-000456",  # hear sad songs
            "massive_domain-test-000914",  # title of song
            "massive_domain-test-000895",  # please skip two songs
            "massive_domain-test-000814",  # play song by van halen
            "massive_domain-test-000026",  # i like this song
            "massive_domain-test-000653",  # rate current song five stars
            "massive_domain-test-000669",  # play me last year's hits
            "massive_domain-test-000899",  # play madonna song
            "massive_domain-test-000187",  # do not play rock metal
            "massive_domain-test-000322",  # add song to running list
            "massive_domain-test-000898",  # play latest song from abbas album
            "massive_domain-test-000372",  # play ella fitzgerald get happy
            "massive_domain-test-000192",  # listen to country music
            "massive_domain-test-001029",  # play bohemian rhapsody
            "massive_domain-test-000469",  # play jazz music
            "massive_domain-test-000647",  # play youtube playlist
            "massive_domain-test-000731",  # play sleepyhead by passion pit
            "massive_domain-test-000435",  # play spotify dance playlist
            "massive_domain-test-000002",  # pink is all we need
            "massive_domain-test-000458",  # hear purple haze
            "massive_domain-test-000283",  # play music released 1990-2000
            "massive_domain-test-000525",  # play keane hopes and fears album
            "massive_domain-test-000676",  # play shape of you by ed sheeran
            "massive_domain-test-000942",  # music
            "massive_domain-test-000290",  # change music mode to rock
            "massive_domain-test-000241",  # play latest rock songs 4+ stars
            "massive_domain-test-000541",  # i like jazz and hate disco
            "massive_domain-test-001018",  # replay the musics
            "massive_domain-test-001023",  # play good song from playlist
            "massive_domain-test-000401",  # song info
            "massive_domain-test-001620",  # play
            "massive_domain-test-000779",  # play song next
            "massive_domain-test-001912",  # what is on my playlist
        ],
        "reasoning": "Music playback service: play/skip/rate songs, choose genres, manage playlists, song metadata.",
    },
    {
        "name": "Radio / podcast / audiobook service",
        "description": "Spoken-audio / streaming-radio backing service: live radio stations, podcasts, audiobooks (distinct from on-demand music tracks).",
        "text_ids": [
            "massive_domain-test-001643",  # play any pop fm channel
            "massive_domain-test-002021",  # play favorite podcast from list
            "massive_domain-test-002001",  # play the podcast for me
            "massive_domain-test-000980",  # what's on the radio right now
            "massive_domain-test-001990",  # play me john's podcast
            "massive_domain-test-001635",  # station that plays r&b
            "massive_domain-test-001634",  # start radio play
            "massive_domain-test-001667",  # start my playlist on iheart radio
            "massive_domain-test-001988",  # shadi special podcast play it
            "massive_domain-test-001683",  # resume harry potter book four
            "massive_domain-test-001649",  # hear the bob cesca show
            "massive_domain-test-001664",  # help listening to the radio
            "massive_domain-test-001989",  # previous episode of podcast
            "massive_domain-test-001971",  # play young turks podcast
            "massive_domain-test-001642",  # radio station playing howard stern
            "massive_domain-test-001599",  # turn on the radio
        ],
        "reasoning": "Distinct backing service for live-radio tuning, podcast playback and audiobook resume — different content APIs from music libraries.",
    },
    {
        "name": "Weather service",
        "description": "Weather backing service: forecasts, current conditions, temperature trends, clothing/jacket suggestions tied to weather.",
        "text_ids": [
            "massive_domain-test-000952",  # take a sun glass
            "massive_domain-test-000902",  # weather to expect this week
            "massive_domain-test-000326",  # fall in temperature tonight
            "massive_domain-test-000131",  # summarize week's weather
            "massive_domain-test-000585",  # weather call for rain saturday
            "massive_domain-test-001076",  # should i wear a jacket
            "massive_domain-test-000260",  # weather for my location this week
            "massive_domain-test-001078",  # how's the weather today
            "massive_domain-test-000661",  # weather in orange tx
            "massive_domain-test-000449",  # is it raining tonight
            "massive_domain-test-000823",  # do i need a jacket
            "massive_domain-test-000877",  # what's the weather now
            "massive_domain-test-000574",  # what is the ten day forecast
            "massive_domain-test-000060",  # weather in barcelona in two days
            "massive_domain-test-000875",  # how cold is today
            "massive_domain-test-001002",  # change car tires to snow tires soon
        ],
        "reasoning": "Weather API service powers forecast/conditions/temperature lookups and clothing suggestions tied to outside conditions.",
    },
    {
        "name": "Alarm service",
        "description": "Alarm-clock backing service: set, list, modify, or remove alarms, including wake-me-up requests.",
        "text_ids": [
            "massive_domain-test-000571",  # set alarm for 5 pm
            "massive_domain-test-000108",  # set alarm for 9 am
            "massive_domain-test-000881",  # make a new alarm
            "massive_domain-test-000284",  # remove tuesday alarm
            "massive_domain-test-000953",  # get me up at six am
            "massive_domain-test-000666",  # list current set alarms
            "massive_domain-test-000292",  # set alarm monday morning 6am
            "massive_domain-test-000938",  # set alarm six am
            "massive_domain-test-000655",  # remove weekdays 9am alarm
            "massive_domain-test-000013",  # tell me about my alarms
            "massive_domain-test-000079",  # remove the first alarm
            "massive_domain-test-000867",  # what times are my alarms set for
            "massive_domain-test-000248",  # what alarms did i set
            "massive_domain-test-000968",  # set alarm at 9am next sunday
        ],
        "reasoning": "Dedicated alarm service handles create/list/remove operations on alarms keyed to wake-up times.",
    },
    {
        "name": "Calendar / appointments service",
        "description": "Calendar backing service: schedule, list, modify, or remove events, meetings, appointments tied to a calendar date.",
        "text_ids": [
            "massive_domain-test-001839",  # remove car payment on calendar
            "massive_domain-test-001408",  # do i have a sales meeting today
            "massive_domain-test-001881",  # remove dentist appointment
            "massive_domain-test-001556",  # make event
            "massive_domain-test-001481",  # add event to calendar app
            "massive_domain-test-001516",  # is jessica's birthday on april 12
            "massive_domain-test-001554",  # important meetings with my boss over the week
            "massive_domain-test-001288",  # mark calendar - unavailable tomorrow
            "massive_domain-test-001328",  # set meeting to discuss terrorism with fred
            "massive_domain-test-001482",  # meeting reminders 3 to 5
            "massive_domain-test-001563",  # remove medical appointment from calendar
            "massive_domain-test-001327",  # schedule meeting with derrick at noon
            "massive_domain-test-001486",  # add meeting to calendar for 9am tom
            "massive_domain-test-001259",  # get rid of events on the 19th
            "massive_domain-test-001531",  # remind upcoming meeting with eminem
            "massive_domain-test-001384",  # remove appointment with doctor saturday
            "massive_domain-test-001552",  # meetings scheduled today
            "massive_domain-test-001451",  # what is going on on december 4th
            "massive_domain-test-001576",  # exhibition 2017 mass mar 25 note on date
            "massive_domain-test-001353",  # add practice on feb 4 kings park 2pm
        ],
        "reasoning": "Calendar service: events, meetings, appointments tied to specific dates and people; create/list/remove operations.",
    },
    {
        "name": "Reminders / to-do service",
        "description": "Reminders backing service: time- or task-based reminders without a fixed calendar slot — nag me, remind me, what is my next reminder.",
        "text_ids": [
            "massive_domain-test-001393",  # send an alert before meeting
            "massive_domain-test-001378",  # remind take out garbage at 6pm
            "massive_domain-test-001470",  # set reminder daughter's birthday
            "massive_domain-test-001550",  # remind me two days before wife birthday
            "massive_domain-test-001200",  # remind me to wash the windows
            "massive_domain-test-001494",  # remind water plants tu/th/sat
            "massive_domain-test-001455",  # laundry 8pm create reminder
            "massive_domain-test-001202",  # what is my next reminder
            "massive_domain-test-001222",  # set reminder print docs tuesday morning
            "massive_domain-test-000271",  # new topic on politics, alert me
            "massive_domain-test-001389",  # set reminder two days prior of event
            "massive_domain-test-002266",  # remind me to move so no weight gain
            "massive_domain-test-001863",  # what is on my to do list today
            "massive_domain-test-001529",  # clear my next activity
        ],
        "reasoning": "Reminders/to-do service: ad-hoc nudges, alerts, and to-do queries separate from formal calendar events.",
    },
    {
        "name": "Lists / shopping-list service",
        "description": "Generic list backing service: shopping lists, to-do lists, contact lists — create, add, remove, read list contents.",
        "text_ids": [
            "massive_domain-test-001857",  # add red wine to shopping list
            "massive_domain-test-001893",  # clear the list
            "massive_domain-test-001879",  # tell me that list i wrote two days ago
            "massive_domain-test-001858",  # remove the list called party time
            "massive_domain-test-001837",  # make this week's shopping list
            "massive_domain-test-001946",  # please create new list
            "massive_domain-test-001917",  # what groups are listed in my contacts
            "massive_domain-test-001936",  # remove milk from shopping list
            "massive_domain-test-001937",  # get rid of tax list from 1990
            "massive_domain-test-001849",  # reset my locations list
            "massive_domain-test-001894",  # delete list for groceries
        ],
        "reasoning": "List manager service: shopping/contact/generic lists with CRUD operations.",
    },
    {
        "name": "Smart-home service",
        "description": "Smart-home backing service: lights, plugs/sockets, vacuums, coffee makers, HVAC — control physical devices in the home.",
        "text_ids": [
            "massive_domain-test-000122",  # please start vacuum cleaner
            "massive_domain-test-000383",  # start coffee at 6 am
            "massive_domain-test-000379",  # light color for study room
            "massive_domain-test-000396",  # put dark colors instead of light in house
            "massive_domain-test-000787",  # change color of the lights
            "massive_domain-test-000933",  # low cooling condition
            "massive_domain-test-000413",  # start robot vacuum cleaner
            "massive_domain-test-000858",  # turn off my wemo plug
            "massive_domain-test-000864",  # turn off sockets
            "massive_domain-test-000870",  # make the room brighter
            "massive_domain-test-000626",  # different color in the kitchen
            "massive_domain-test-000515",  # shut off the socket
            "massive_domain-test-000262",  # turn on wemo
            "massive_domain-test-000086",  # turn off the lights in the bathroom
            "massive_domain-test-000240",  # i'd like some red lighting
            "massive_domain-test-000382",  # change house lights color to blue
            "massive_domain-test-000397",  # put dark colors instead of light ones
            "massive_domain-test-000495",  # start the vacuum
        ],
        "reasoning": "IoT/smart-home service controlling lights, plugs, vacuums, HVAC and other home devices.",
    },
    {
        "name": "Email service",
        "description": "Email backing service: send, read, reply, check inbox, alert on new mail.",
        "text_ids": [
            "massive_domain-test-002932",  # say to email george urgent money
            "massive_domain-test-002872",  # locate email to giant eagle, add to contacts
            "massive_domain-test-002859",  # email to comcast about service issues
            "massive_domain-test-002745",  # send email to john about meeting
            "massive_domain-test-002874",  # reply to sarah's email
            "massive_domain-test-002799",  # do i have new emails
            "massive_domain-test-002885",  # are there new emails today
            "massive_domain-test-002708",  # anything new in my mailbox
            "massive_domain-test-002818",  # read new emails
            "massive_domain-test-002804",  # send email to margaret
            "massive_domain-test-002940",  # email mom ask weather
            "massive_domain-test-002787",  # pull up kate's email write reply
            "massive_domain-test-002786",  # load new email for bruce and send
            "massive_domain-test-002952",  # show emails received last hour
            "massive_domain-test-002792",  # send email to joe at aol
            "massive_domain-test-002957",  # read list of new emails
            "massive_domain-test-002947",  # any new emails after 4pm today
            "massive_domain-test-002825",  # show emails from my boss
            "massive_domain-test-002920",  # check my gmail for new mail
            "massive_domain-test-002751",  # send email to rohit i am busy
            "massive_domain-test-002962",  # please read my new mails
            "massive_domain-test-002899",  # beep when i get email from john
            "massive_domain-test-002740",  # subject of last email from mom
            "massive_domain-test-002853",  # that last email needs answer asap
            "massive_domain-test-002942",  # send email to tim at hotmail
            "massive_domain-test-002758",  # send email to my family
        ],
        "reasoning": "Email backing service: full CRUD on a mailbox plus send/reply and inbox alerts.",
    },
    {
        "name": "Social-media service",
        "description": "Social-media backing service: post status, tweet, check what's happening on Facebook/Twitter/Instagram, social notifications.",
        "text_ids": [
            "massive_domain-test-002619",  # post status on facebook re weather
            "massive_domain-test-002661",  # send complaint via tweet to j crew
            "massive_domain-test-002603",  # post in twitter visit to japan
            "massive_domain-test-002654",  # tweet complaint for online bookstore
            "massive_domain-test-001185",  # friend updates
            "massive_domain-test-002633",  # who is doing facebook live now
            "massive_domain-test-002625",  # notifications from social media
            "massive_domain-test-002617",  # scan social media tell what's happening
            "massive_domain-test-002621",  # post the message now
            "massive_domain-test-002699",  # set feeling happy status on facebook
            "massive_domain-test-002646",  # what's happening on instagram
            "massive_domain-test-002600",  # birthday wishes
        ],
        "reasoning": "Social posting/feed service spanning Facebook, Twitter, Instagram — distinct from email/SMS messaging.",
    },
    {
        "name": "Complaints / customer-service service",
        "description": "Customer-service backing service: lodge complaints with companies, write reviews, route help requests.",
        "text_ids": [
            "massive_domain-test-002684",  # help complaint to consumer service
            "massive_domain-test-002702",  # submit negative review about a company
            "massive_domain-test-002676",  # open company name find complaints
            "massive_domain-test-002771",  # help is to be sent only to jane
        ],
        "reasoning": "Customer-service / complaints routing service — submit and look up complaints and reviews tied to companies.",
    },
    {
        "name": "Contacts / phone-book service",
        "description": "Contacts backing service: look up phone numbers and contact details.",
        "text_ids": [
            "massive_domain-test-002261",  # do you know jessica snout's phone number
        ],
        "reasoning": "Contact-directory service: phone-number and personal-detail lookups for known people.",
    },
    {
        "name": "Navigation / transit-booking service",
        "description": "Maps / transit-booking backing service: navigation, train/airport bookings, traffic, ETA, local-place directions.",
        "text_ids": [
            "massive_domain-test-002196",  # what is the traffic like now
            "massive_domain-test-002187",  # when is the next train leaving for austin
            "massive_domain-test-002207",  # book ticket to ny by train
            "massive_domain-test-002167",  # find train ticket to philadelphia
            "massive_domain-test-002209",  # book ticket jaipur to mumbai tuesday
            "massive_domain-test-002172",  # lead the way to the place
            "massive_domain-test-002161",  # how long to get to airport
            "massive_domain-test-002208",  # book ticket rajdhani express train
            "massive_domain-test-002251",  # drive the car bring friends over
            "massive_domain-test-002199",  # find train ticket to boston
            "massive_domain-test-002181",  # get ticket via train to orlando
            "massive_domain-test-002119",  # how to get to train station
            "massive_domain-test-002214",  # how is the traffic at the moment
        ],
        "reasoning": "Navigation + transit-booking service: traffic, route guidance, train/airport ticket bookings.",
    },
    {
        "name": "Local-search / yellow-pages service",
        "description": "Local discovery backing service: find nearby shops, restaurants, food courts, events around me.",
        "text_ids": [
            "massive_domain-test-002069",  # what's happening around me
            "massive_domain-test-002044",  # find clothing stores within one mile
            "massive_domain-test-002050",  # reviews of nearest food court
            "massive_domain-test-002079",  # is there a food festival in the area
            "massive_domain-test-000350",  # tell me current events in hometown
            "massive_domain-test-002105",  # what shops are near me
            "massive_domain-test-002268",  # movie theater prices nearby
        ],
        "reasoning": "Local-search / yellow-pages service: find nearby places, events, reviews tied to my location.",
    },
    {
        "name": "Recipes / cooking service",
        "description": "Cooking / recipes backing service: recipes, ingredient substitutions, cooking techniques and times.",
        "text_ids": [
            "massive_domain-test-001718",  # tell me the recipe of
            "massive_domain-test-001728",  # best oven temperature roast potatoes
            "massive_domain-test-001731",  # easiest way to cook pasta
            "massive_domain-test-001729",  # best way to cook pasta al dente
            "massive_domain-test-001754",  # find a recipe for dinner tonight
            "massive_domain-test-001780",  # find recipe for sambar in cookingforu
            "massive_domain-test-001733",  # what conducts heat better
            "massive_domain-test-001765",  # how long do you cook a 15 lb turkey
            "massive_domain-test-001734",  # ingredients for welsh rarebit
            "massive_domain-test-001732",  # ingredient instead of saffron
        ],
        "reasoning": "Recipe/cooking service: recipe lookup, ingredient substitution and cooking techniques.",
    },
    {
        "name": "Food-ordering service",
        "description": "Food-ordering / takeaway backing service: place takeout orders, check order status, takeaway availability.",
        "text_ids": [
            "massive_domain-test-000419",  # call for take-out
            "massive_domain-test-000356",  # hows the food order going
            "massive_domain-test-000029",  # can they provide takeaway
        ],
        "reasoning": "Food-ordering / takeout backing service: place orders, ask about takeaway, check order status.",
    },
    {
        "name": "News service",
        "description": "News backing service: headlines, breaking news, topic-based news (politics, crime, local), entertainment-news (TV show recaps).",
        "text_ids": [
            "massive_domain-test-000130",  # last news from cnn
            "massive_domain-test-000511",  # front page news articles
            "massive_domain-test-002087",  # what movies are out this week
            "massive_domain-test-001138",  # baseball scores
            "massive_domain-test-000308",  # which teams playing in premier league
            "massive_domain-test-000403",  # give me the latest news
            "massive_domain-test-000447",  # latest news from the area
            "massive_domain-test-000784",  # latest crime statistics for my area
            "massive_domain-test-000920",  # bbc news of migration
            "massive_domain-test-000237",  # tell me about trump
            "massive_domain-test-000221",  # foreign policy of trump
            "massive_domain-test-000681",  # what happened on the walking dead
            "massive_domain-test-001015",  # hot topic
            "massive_domain-test-001126",  # technology
            "massive_domain-test-001276",  # a good first impression
        ],
        "reasoning": "News service: politics, sports scores, local/world headlines, entertainment-news such as TV-show recaps.",
    },
    {
        "name": "General knowledge / Q&A service",
        "description": "General-knowledge backing service: encyclopedic facts, definitions, person info, math, philosophical / cosmological questions.",
        "text_ids": [
            "massive_domain-test-000102",  # what's a good joke
            "massive_domain-test-002233",  # highest peak in the world
            "massive_domain-test-002418",  # what year did george clooney start acting
            "massive_domain-test-002298",  # when will the world end
            "massive_domain-test-002232",  # definition of forensic
            "massive_domain-test-002472",  # world population after 20 years
            "massive_domain-test-002533",  # what is a president
            "massive_domain-test-002323",  # why is the earth round
            "massive_domain-test-002495",  # can we go to sun
            "massive_domain-test-002281",  # info about john abraham lincoln
            "massive_domain-test-000229",  # what's a good joke
            "massive_domain-test-002394",  # what flabbergasted means
            "massive_domain-test-002390",  # tell me about wayne gretzky
            "massive_domain-test-002440",  # does kim kardashian wear converse
            "massive_domain-test-002570",  # details about bruce lee
            "massive_domain-test-002299",  # sum of two numbers 4 and 6
            "massive_domain-test-002453",  # can you do nine plus two
            "massive_domain-test-002254",  # how much is one plus one
            "massive_domain-test-002409",  # coordinates of the equator
            "massive_domain-test-002410",  # describe hell
            "massive_domain-test-002584",  # info on lisa ann
            "massive_domain-test-002338",  # where do most celebrities hang out
            "massive_domain-test-002561",  # in which field does that person excel
            "massive_domain-test-002276",  # what does priyanka chopra do
            "massive_domain-test-002711",  # where seth lives
            "massive_domain-test-001140",  # web searches
            "massive_domain-test-001168",  # what do you think about future
        ],
        "reasoning": "General-knowledge Q&A service: facts, definitions, math, biographical info — anything that hits a web/knowledge API.",
    },
    {
        "name": "Finance / stocks-and-currency service",
        "description": "Finance backing service: stock prices, currency exchange rates.",
        "text_ids": [
            "massive_domain-test-002465",  # stock price of hdfc
            "massive_domain-test-002413",  # us dollar and euro exchange rate
            "massive_domain-test-002574",  # exchange rate us vs indian rupees
            "massive_domain-test-002364",  # currency exchange rate for china
            "massive_domain-test-002442",  # price of starbucks stock
            "massive_domain-test-002484",  # exchange rate usd to pound
            "massive_domain-test-002494",  # detail for stock price
            "massive_domain-test-002360",  # exchange rate usd to jpy
        ],
        "reasoning": "Finance service: stock quotes and currency exchange rates.",
    },
    {
        "name": "Clock / time / date service",
        "description": "Clock backing service: current time, date queries, time-zone conversion, calendar-date arithmetic with no event involved.",
        "text_ids": [
            "massive_domain-test-001003",  # a quarter to two
            "massive_domain-test-000636",  # today is which date
            "massive_domain-test-000418",  # is the time currently correct
            "massive_domain-test-000566",  # date for next tuesday
            "massive_domain-test-000700",  # convert time gmt to est
            "massive_domain-test-000986",  # display date
            "massive_domain-test-000937",  # update to current time
            "massive_domain-test-000128",  # time now in new york
            "massive_domain-test-000386",  # it is seven am pacific
            "massive_domain-test-000751",  # hours till midnight london
            "massive_domain-test-000674",  # say the time
            "massive_domain-test-000207",  # if sunrise stockholm what time okinawa
        ],
        "reasoning": "Time/date/clock service: tell-time, date arithmetic, time-zone conversion — no alarms or calendars involved.",
    },
    {
        "name": "Games & general-app service",
        "description": "App / games backing service: launch apps, play games like chess, generic device/app controls (volume, open folder).",
        "text_ids": [
            "massive_domain-test-000177",  # open the folder app please
            "massive_domain-test-001815",  # play chess with me
            "massive_domain-test-001794",  # play temple run game
            "massive_domain-test-001010",  # decrease twenty percent
            "massive_domain-test-000276",  # turn the volume down
            "massive_domain-test-000974",  # change the volume at
        ],
        "reasoning": "Game and generic-app launcher / device-control service — chess, temple run, volume, open folder.",
    },
    {
        "name": "Assistant chit-chat / meta service",
        "description": "The assistant's own conversational/meta service: greetings, journal-style life updates, can't-hear feedback, generic chit-chat with no external API.",
        "text_ids": [
            "massive_domain-test-001139",  # what i like today
            "massive_domain-test-001083",  # please see what you wrote for this question
            "massive_domain-test-001084",  # i don't know how to answer
            "massive_domain-test-001093",  # i had food as soon as i got up
            "massive_domain-test-001292",  # what's for today
            "massive_domain-test-001096",  # met one of old classmates today
            "massive_domain-test-001105",  # my day was extremely hard
            "massive_domain-test-001092",  # lets have a chat
            "massive_domain-test-000047",  # i can barely hear you
            "massive_domain-test-001085",  # i am going to work today
            "massive_domain-test-002671",  # clear out the problem
            "massive_domain-test-001086",  # i had a bad day
            "massive_domain-test-000008",  # what's up olly
            "massive_domain-test-000995",  # opinion petabit
        ],
        "reasoning": "Chit-chat / journal / self-feedback handled by the assistant itself without any external service backend.",
    },
]

# Verify coverage.
assigned = []
for c in clusters:
    assigned.extend(c["text_ids"])

all_ids = [x["id"] for x in items]
assigned_set = set(assigned)
all_set = set(all_ids)

# Check duplicates
dupes = [x for x in assigned if assigned.count(x) > 1]
if dupes:
    print("DUPLICATES:", set(dupes))
    raise SystemExit(1)

missing = all_set - assigned_set
extra = assigned_set - all_set

print("k =", len(clusters))
print("total assigned:", len(assigned))
print("unique assigned:", len(assigned_set))
print("missing:", len(missing))
if missing:
    for m in sorted(missing):
        t = next(it["text"] for it in items if it["id"] == m)
        print("  MISSING:", m, "|", t)
print("extras (not in sample):", len(extra))
if extra:
    for e in sorted(extra):
        print("  EXTRA:", e)

if missing or extra or dupes:
    raise SystemExit("coverage error — fix before writing")

# Build proposal payload
ts = datetime.now()
proposal = {
    "timestamp": ts.isoformat(),
    "sample_size": len(items),
    "sample_strategy": "shared_orchestrator_sample",
    "style": "service-provider lens (backing-skill routing)",
    "existing_clusters_considered": False,
    "clusters": clusters,
    "unclustered_ids": [],
    "observations": (
        "Viewed each utterance as a routing decision: which backing service / skill "
        "would the voice assistant dispatch this to. Yielded 22 clusters within the 14-22 target band. "
        "Music vs. radio/podcast were kept separate because their content APIs differ (on-demand "
        "track playback vs. live tuning / spoken-audio streams). Alarms, reminders, calendar and lists "
        "are also separated even though they cluster as 'productivity' under a coarser lens — they hit "
        "distinct backend services with different data models. Email and social-media messaging are "
        "split for the same reason. The 'Assistant chit-chat / meta' cluster covers utterances with no "
        "external backing service (journal-style updates, self-feedback, greetings); per task constraint, "
        "no 'other'/'misc' was used. Borderline calls: 'shall i change my car tires to snow tires soon' "
        "→ Weather (it's a weather-driven decision); 'what movies are out this week' → News (entertainment "
        "headlines, no booking); 'movie theater prices nearby' → Local-search; 'baseball scores' / "
        "'premier league' → News (sports-news service)."
    ),
}

dt_str = ts.strftime("%Y%m%d_%H%M%S")
uid = uuid.uuid4().hex[:4]
out_path = WORKSPACE / "proposals" / f"prop_{dt_str}_{uid}.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(proposal, f, indent=2)

print("WROTE:", out_path)
