"""Build a USER-INTENT-CLASS lens proposal for the 300-text auditdiv sample.

Lens: group utterances by the high-level *kind* of interaction the user is
initiating (their goal/speech-act), not by the backend application or topic.
"""
import json
import uuid
from datetime import datetime
from pathlib import Path

WS = Path(r'C:\Users\emily\Documents\agentic-clustering\results\clustering\massive_domain\seed=0_auditdiv')
with open(WS / '_shared_sample.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

items = {d['id']: d['text'] for d in data}

# -----------------------------------------------------------------------------
# Cluster definitions (USER-INTENT-CLASS lens)
# -----------------------------------------------------------------------------
# A1  factual-information-retrieval     — ask the assistant for a fact / lookup
# A2  status-state-query                — ask about state of an existing item/setting (alarms, lists, traffic)
# A3  contextual-lookup-current-info    — weather/time/news/traffic NOW
# A4  computation-conversion-definition — math, unit/currency conversion, definitions
# B1  device-iot-control                — actuate smart-home devices/sockets/appliances
# B2  media-playback-control            — play/skip/resume music/podcast/radio/games
# B3  media-volume-mode-settings        — change volume / mode / rate / song info during playback
# C1  scheduling-create                 — create alarms / reminders / calendar events / lists
# C2  scheduling-modify-delete          — modify, remove or query existing alarms/events/lists
# D1  communication-outbound            — send email/tweet/sms/post (assistant authoring on user's behalf)
# D2  communication-inbound-check       — read/check incoming email/social notifications
# D3  social-monitoring                 — view what others are posting / social feed scanning
# E1  navigation-transit-booking        — get directions, find shops, book train/transit
# E2  recipe-howto-instruction          — request a procedure / recipe / how-to
# F1  smalltalk-personal-disclosure     — chit-chat, jokes, complaints about day, personal statements
# G1  shopping-list-management          — explicit lists (shopping/to-do/notes) excluded from C
# H1  malformed-or-fragmentary          — single tokens / unparseable / failed turn-takings

# We will assign each text id to exactly one cluster, no 'misc'.

assignments = {}

# Helper to bulk-assign
def assign(ids, cluster):
    for i in ids:
        assert i in items, f"unknown id {i}"
        assert i not in assignments, f"double-assign {i} -> {cluster} (already {assignments[i]})"
        assignments[i] = cluster

# -----------------------------------------------------------------------------
# A1  factual-information-retrieval (asking for a fact / lookup about world or person)
# -----------------------------------------------------------------------------
A1 = [
    'massive_domain-test-002233',  # highest peak in the world
    'massive_domain-test-002418',  # what year did george clooney start acting
    'massive_domain-test-002298',  # when will the world end
    'massive_domain-test-002232',  # definition of forensic  -> def -> A4 actually
    'massive_domain-test-002472',  # world population after twenty years
    'massive_domain-test-002281',  # info about john abraham lincoln
    'massive_domain-test-002323',  # why is the earth round
    'massive_domain-test-002390',  # tell me about wayne gretzky
    'massive_domain-test-002394',  # what flabbergasted means -> A4 def
    'massive_domain-test-002440',  # does kim kardashian wear converse
    'massive_domain-test-002533',  # what is a president
    'massive_domain-test-002409',  # coordinates of the equator
    'massive_domain-test-002410',  # describe hell
    'massive_domain-test-002338',  # where do most celebrities hang out
    'massive_domain-test-002561',  # in which field does that person excel in
    'massive_domain-test-002570',  # details about bruce lee
    'massive_domain-test-002584',  # info on lisa ann
    'massive_domain-test-002276',  # what does priyanka chopra do to look beautiful
    'massive_domain-test-002711',  # where seth lives
    'massive_domain-test-000237',  # tell me about trump
    'massive_domain-test-000221',  # foreign policy of trump  -> news? more like factual about a topic
    'massive_domain-test-002495',  # can we go to sun
    'massive_domain-test-001168',  # what do you think about future  -> smalltalk? more like opinion-y -> F1
]
# Refine: move definitions/conversions to A4
A1 = [x for x in A1 if x not in {
    'massive_domain-test-002232',
    'massive_domain-test-002394',
    'massive_domain-test-001168',
    'massive_domain-test-000221',
}]
# A1 still: world-fact / person-fact / philosophical-question type queries
assign(A1, 'A1')

# -----------------------------------------------------------------------------
# A2  status-state-query (state of existing items the assistant manages: alarms/lists/calendar/email-count etc.)
# -----------------------------------------------------------------------------
A2 = [
    'massive_domain-test-000356',  # hows the food order going
    'massive_domain-test-000666',  # list current set alarms
    'massive_domain-test-001408',  # do i have a sales meeting today
    'massive_domain-test-001516',  # is jessica's birthday on april twelfth
    'massive_domain-test-001917',  # what groups are listed in my contacts
    'massive_domain-test-000867',  # what times are my alarms set for
    'massive_domain-test-001202',  # what is my next reminder
    'massive_domain-test-001879',  # tell me that list i wrote two days ago
    'massive_domain-test-001482',  # meeting reminders from three to five
    'massive_domain-test-001531',  # remind upcoming meeting with eminem  (state query of upcoming meeting)
    'massive_domain-test-001554',  # just let me know the important meetings with my boss over the week
    'massive_domain-test-001863',  # what is on my to do list today
    'massive_domain-test-001912',  # what is on my playlist
    'massive_domain-test-000401',  # song info  -> media-info? -> B3 actually
    'massive_domain-test-000013',  # tell me about my alarms
    'massive_domain-test-000248',  # what alarms did i set
    'massive_domain-test-001552',  # what are meeting scheduled for today
    'massive_domain-test-001451',  # what is going on on december fourth (calendar lookup)
    'massive_domain-test-002261',  # do you know jessica snout's phone number (contact info)
    'massive_domain-test-001083',  # please see what you wrote for this question (assistant self-state)
]
# move song info to B3
A2 = [x for x in A2 if x != 'massive_domain-test-000401']
assign(A2, 'A2')

# -----------------------------------------------------------------------------
# A3  contextual-lookup-current-info (weather/news/time/traffic/stocks/exchange/finance/scores)
# -----------------------------------------------------------------------------
A3 = [
    'massive_domain-test-000130',  # last news from cnn
    'massive_domain-test-002069',  # what's happening around me
    'massive_domain-test-002465',  # stock price of hdfc
    'massive_domain-test-000902',  # what kind of weather should i expect this week
    'massive_domain-test-002413',  # us dollar and euro exchange rate
    'massive_domain-test-002196',  # what is the traffic like now
    'massive_domain-test-000511',  # front page news articles please
    'massive_domain-test-002574',  # prevailing exchange rate us versus indian rupees
    'massive_domain-test-002364',  # currency exchange rate for china
    'massive_domain-test-000326',  # will there be a fall in temperature by tonight
    'massive_domain-test-000131',  # summarize week's weather details
    'massive_domain-test-000585',  # does the weather call for rain saturday
    'massive_domain-test-000260',  # weather for my location this week
    'massive_domain-test-002442',  # price of starbuck's stock
    'massive_domain-test-001078',  # how's the weather today
    'massive_domain-test-000661',  # what kind of weather are they having in orange tx right now
    'massive_domain-test-001138',  # baseball scores
    'massive_domain-test-000636',  # today is which date
    'massive_domain-test-000418',  # is the time currently correct for where i am located
    'massive_domain-test-000700',  # convert current time gmt to est  -> A4 conversion? but it's time-current -> A4
    'massive_domain-test-002187',  # when is the next train leaving for austin
    'massive_domain-test-002409',  # coordinates of equator -> nope already A1
    'massive_domain-test-002494',  # detail for stock price
    'massive_domain-test-000128',  # what is the time now in new york
    'massive_domain-test-000937',  # update to current time
    'massive_domain-test-000877',  # what's the weather now
    'massive_domain-test-002360',  # list exchange rate info for the day
    'massive_domain-test-002484',  # find exchange rate usd to pound
    'massive_domain-test-000986',  # display date
    'massive_domain-test-000674',  # say the time
    'massive_domain-test-002214',  # how is the traffic at the moment
    'massive_domain-test-000060',  # weather in barcelona in two days
    'massive_domain-test-000875',  # how cold is today
    'massive_domain-test-000566',  # what is the date for next tuesday
    'massive_domain-test-002087',  # what movies are out this week
    'massive_domain-test-000308',  # which teams are playing today in the premier league
    'massive_domain-test-000681',  # what happened on the walking dead
    'massive_domain-test-000823',  # do i need a jacket  -> weather inference
    'massive_domain-test-001076',  # should i wear a jacket -> weather inference
    'massive_domain-test-001002',  # shall i change my car tires to snow tires soon -> weather inference
    'massive_domain-test-000449',  # is it raining tonight
    'massive_domain-test-000350',  # tell me all the current events in my hometown
    'massive_domain-test-000447',  # latest news from the area
    'massive_domain-test-000784',  # latest crime statistics for me area
    'massive_domain-test-000574',  # ten day forecast
    'massive_domain-test-000751',  # in how many hours will it be midnight in london england  -> time conversion? -> A4 then
    'massive_domain-test-000207',  # if it's sunrise in stockholm what time is it in okinawa -> time conversion -> A4
    'massive_domain-test-000386',  # it is seven am in pacific standard time -> user statement of time -> F1? more like time-state -> A3
    'massive_domain-test-000403',  # give me the latest news
    'massive_domain-test-000920',  # show bbc news of migration
]
# Move pure conversions to A4
A3 = [x for x in A3 if x not in {
    'massive_domain-test-000700',  # gmt -> est is conversion
    'massive_domain-test-000751',
    'massive_domain-test-000207',
}]
# 002409 not in A3 anyway
assign([x for x in A3 if x != 'massive_domain-test-002409'], 'A3')

# -----------------------------------------------------------------------------
# A4  computation-conversion-definition (math/units/currency-convert/definitions)
# -----------------------------------------------------------------------------
A4 = [
    'massive_domain-test-002232',  # definition of forensic
    'massive_domain-test-002394',  # what flabbergasted means
    'massive_domain-test-002299',  # sum of four and six
    'massive_domain-test-002453',  # nine plus two
    'massive_domain-test-002254',  # how much is one plus one
    'massive_domain-test-000700',  # convert current time gmt to est
    'massive_domain-test-000751',  # hours until midnight london
    'massive_domain-test-000207',  # sunrise stockholm what time okinawa
    'massive_domain-test-001733',  # what conducts heat better copper bottomed pots or cast iron -> general knowledge -> A1 maybe, but it's a factual lookup; could go A1; place in A1 since it's a factual question
]
# Move 001733 back to A1; it's a fact question, not a conversion
A4 = [x for x in A4 if x != 'massive_domain-test-001733']
assign(A4, 'A4')
assign(['massive_domain-test-001733'], 'A1')

# -----------------------------------------------------------------------------
# B1  device-iot-control (smart-home / sockets / lights / appliances)
# -----------------------------------------------------------------------------
B1 = [
    'massive_domain-test-000122',  # please start vacuum cleaner
    'massive_domain-test-000383',  # start coffee at six am  -> actually a scheduled action? but at-six-am means starts then; treat as device-control with schedule = B1
    'massive_domain-test-000787',  # change color of the lights
    'massive_domain-test-000379',  # light color for study room
    'massive_domain-test-000396',  # put dark colors in the house
    'massive_domain-test-000858',  # turn off my wemo plug
    'massive_domain-test-000864',  # turn off sockets
    'massive_domain-test-000870',  # can you make the room brighter
    'massive_domain-test-000262',  # turn on wemo
    'massive_domain-test-000626',  # like in the kitchen to be a different color
    'massive_domain-test-000086',  # turn off the lights in the bathroom
    'massive_domain-test-000413',  # start robot vacuum cleaner
    'massive_domain-test-000495',  # start the vacuum
    'massive_domain-test-000515',  # shut off the socket
    'massive_domain-test-000933',  # low cooling condition  -> ambiguous; treat as IoT setting -> B1
    'massive_domain-test-000382',  # change house lights color to blue
    'massive_domain-test-000240',  # red lighting
    'massive_domain-test-001010',  # decrease twenty percent  -> ambiguous; could be volume; treat as device/setting -> B3 volume mode? Actually unclear; put it in B3
    'massive_domain-test-000397',  # olly put dark colors instead of light ones  -> B1
    'massive_domain-test-000177',  # open the folder app please  -> system actuation: opening an app
]
# move 001010 to B3 (volume-ish ambiguous setting change)
B1 = [x for x in B1 if x != 'massive_domain-test-001010']
assign(B1, 'B1')

# -----------------------------------------------------------------------------
# B2  media-playback-control (play/skip/resume music/podcast/radio/audiobook/game)
# -----------------------------------------------------------------------------
B2 = [
    'massive_domain-test-000456',  # i need to hear some sad songs today
    'massive_domain-test-000814',  # play me a song by van halen
    'massive_domain-test-000895',  # please skip two songs
    'massive_domain-test-000372',  # play ella fitzgerald get happy
    'massive_domain-test-000192',  # listen to country music
    'massive_domain-test-001643',  # play any pop f. m. channel
    'massive_domain-test-001620',  # play
    'massive_domain-test-001029',  # play bohemian raphsody
    'massive_domain-test-000469',  # please play some jazz
    'massive_domain-test-000647',  # play youtube playlist blank
    'massive_domain-test-000731',  # play sleepyhead by passion pit
    'massive_domain-test-000435',  # play my spotify dance playlist
    'massive_domain-test-000898',  # play latest song from album abbas
    'massive_domain-test-000899',  # play song from madonna
    'massive_domain-test-000458',  # purple haze
    'massive_domain-test-002001',  # please play the podcast for me
    'massive_domain-test-001990',  # play john's podcast
    'massive_domain-test-001988',  # shadi special podcast play it
    'massive_domain-test-001971',  # play the young turks podcast
    'massive_domain-test-001023',  # play some good song from my play list
    'massive_domain-test-002021',  # play my favorite podcast
    'massive_domain-test-001664',  # listening to the radio
    'massive_domain-test-001599',  # turn on the radio
    'massive_domain-test-001642',  # find the radio station playing howard stern
    'massive_domain-test-000980',  # what's on the radio right now -> info, but radio status -> A2/B3? treat as B3 media-info-now (current playing). I'll put in B3
    'massive_domain-test-000241',  # play latest rock songs rating four or above
    'massive_domain-test-001649',  # bob cesca show
    'massive_domain-test-001683',  # resume harry potter book four
    'massive_domain-test-001815',  # play chess with me
    'massive_domain-test-001794',  # play temple run game for me
    'massive_domain-test-000676',  # shape of you ed sheeran
    'massive_domain-test-000283',  # play music between 1990 and 2000
    'massive_domain-test-001018',  # replay the musics
    'massive_domain-test-000779',  # play song next
    'massive_domain-test-000942',  # music
    'massive_domain-test-001635',  # hear a station that plays r&b
    'massive_domain-test-000525',  # play something from keane's hopes and fears
    'massive_domain-test-001667',  # start my playlist on i heart radio
    'massive_domain-test-001634',  # start radio play
    'massive_domain-test-001989',  # go back to previous episode of podcast
    'massive_domain-test-000187',  # do not play rock metal
    'massive_domain-test-000669',  # play me last year's hits
]
# move 980 to B3
B2 = [x for x in B2 if x != 'massive_domain-test-000980']
assign(B2, 'B2')

# -----------------------------------------------------------------------------
# B3  media-volume-mode-settings (volume, mode/genre change, rate, song info, queue add)
# -----------------------------------------------------------------------------
B3 = [
    'massive_domain-test-000401',  # song info
    'massive_domain-test-000980',  # what's on the radio right now
    'massive_domain-test-000653',  # please rate current song as five stars
    'massive_domain-test-000322',  # add song to running list (music playlist add)
    'massive_domain-test-000290',  # change the music mode to rock
    'massive_domain-test-000276',  # please turn the volume down
    'massive_domain-test-000974',  # can you change the volume at
    'massive_domain-test-000541',  # please note i like jazz and hate disco  -> a preference note for media; could be lists; treat as media-preference -> B3
    'massive_domain-test-000914',  # title of song
    'massive_domain-test-000026',  # i like this song -> media-feedback -> B3
    'massive_domain-test-001010',  # decrease twenty percent
    'massive_domain-test-000047',  # i can barely hear you  -> volume/audio complaint -> B3
]
assign(B3, 'B3')

# -----------------------------------------------------------------------------
# C1  scheduling-create (set alarm, create reminder, add calendar event, create list)
# -----------------------------------------------------------------------------
C1 = [
    'massive_domain-test-000571',  # set my alarm for five pm
    'massive_domain-test-000108',  # set an alarm for nine am
    'massive_domain-test-000881',  # make a new alarm
    'massive_domain-test-001378',  # remind me to take out the garbage
    'massive_domain-test-001470',  # set a reminder for my daughter's birthday
    'massive_domain-test-001393',  # send an alert before meeting
    'massive_domain-test-001556',  # make event
    'massive_domain-test-001550',  # remind me two days before my wife birthday
    'massive_domain-test-001200',  # remind me to wash the windows
    'massive_domain-test-001481',  # add event to calendar app
    'massive_domain-test-001494',  # remind me to water my plants every tuesday thursday saturday
    'massive_domain-test-001455',  # i want to do laundry at eight pm create reminder
    'massive_domain-test-000292',  # set alarm only monday morning six am
    'massive_domain-test-000938',  # set alarm six am
    'massive_domain-test-000953',  # i need you to get me up at six am
    'massive_domain-test-000271',  # once a new topic on politics comes up alert me
    'massive_domain-test-001288',  # i am unavailable from four to six tomorrow mark my calendar
    'massive_domain-test-002899',  # beep when i get an email from john
    'massive_domain-test-002266',  # remind me to move so there is no weight gain
    'massive_domain-test-001222',  # set reminder to print the documents on tuesday
    'massive_domain-test-001486',  # add a meeting to my calendar
    'massive_domain-test-001328',  # please set a meeting to discuss terrorism with fred
    'massive_domain-test-001327',  # please schedule a meeting with derrick in haysi at noon
    'massive_domain-test-001353',  # please add practice on feb four at king's park
    'massive_domain-test-001389',  # set a reminder for two days prior of the event
    'massive_domain-test-000968',  # set alarm at nine am on next sunday
    'massive_domain-test-001576',  # exhibition 2017 mass on mar 25 make a note of it
    'massive_domain-test-001837',  # i want to make this week's shopping list -> create list -> G1 actually since it's a shopping list
    'massive_domain-test-001946',  # please create new list -> G1
    'massive_domain-test-001857',  # add red wine to my shopping list -> G1
]
# move list creates to G1
G1_from_C1 = {'massive_domain-test-001837', 'massive_domain-test-001946', 'massive_domain-test-001857'}
C1 = [x for x in C1 if x not in G1_from_C1]
assign(C1, 'C1')

# -----------------------------------------------------------------------------
# C2  scheduling-modify-delete (remove/clear/modify alarms, calendar events, reminders)
# -----------------------------------------------------------------------------
C2 = [
    'massive_domain-test-001839',  # remove my car payment on my calendar
    'massive_domain-test-001881',  # remove my dentist's appointment from today's schedule
    'massive_domain-test-000284',  # remove tuesday alarm of nine am
    'massive_domain-test-001563',  # remove from calendar my medical appointment
    'massive_domain-test-001384',  # remove appointment with my doctor on saturday
    'massive_domain-test-000655',  # remove the alarm that is set for weekdays at nine
    'massive_domain-test-000079',  # remove the first alarm
    'massive_domain-test-001259',  # get rid of events on the nineteenth from my calendar
    'massive_domain-test-001529',  # clear my next activity
]
assign(C2, 'C2')

# -----------------------------------------------------------------------------
# D1  communication-outbound (send email/tweet/text/post; compose; reply)
# -----------------------------------------------------------------------------
D1 = [
    'massive_domain-test-002619',  # post a new status on facebook
    'massive_domain-test-002771',  # help is to be sent only to jane  -> ambiguous, looks like specifying recipient for messaging -> D1
    'massive_domain-test-002932',  # email to george saying i need the money
    'massive_domain-test-002661',  # send complaint via tweet to j crew
    'massive_domain-test-002859',  # email to comcastcom about my service issues
    'massive_domain-test-002603',  # post in twitter my visit to japan
    'massive_domain-test-002745',  # email to john asking him what time works for the meeting
    'massive_domain-test-002874',  # reply to sarah's email
    'massive_domain-test-002654',  # tweet a complaint for online bookstore
    'massive_domain-test-002684',  # help complaint to consumer service
    'massive_domain-test-002940',  # email mom and ask how the weather is there
    'massive_domain-test-002804',  # send an email to margaret
    'massive_domain-test-002787',  # pull up kate's email and write that i will let her know
    'massive_domain-test-002786',  # please load new email for bruce and send message now
    'massive_domain-test-002751',  # email to rohit saying i am busy tomorrow
    'massive_domain-test-002702',  # submit a negative review about a company
    'massive_domain-test-002621',  # post the message now
    'massive_domain-test-002699',  # set feeling happy status on facebook
    'massive_domain-test-002792',  # send email to joe at aol dot com
    'massive_domain-test-002758',  # i want to send an email to my family do you help me
    'massive_domain-test-002676',  # open company name and find complaints -> ambiguous but looks like outbound complaint flow -> D1
    'massive_domain-test-002853',  # that last email needs to be answer asap  -> reply -> D1
    'massive_domain-test-002942',  # send email to tim at hotmail dot com
    'massive_domain-test-002600',  # birthday wishes  -> ambiguous; could be outbound message; D1
    'massive_domain-test-002872',  # locate the email to giant eagle and put it into my contact list -> contact mgmt; place in A2? It's modifying contacts; I'll put in D1 as email-management
]
assign(D1, 'D1')

# -----------------------------------------------------------------------------
# D2  communication-inbound-check (read/check incoming email / new mail / read social)
# -----------------------------------------------------------------------------
D2 = [
    'massive_domain-test-002885',  # are their any new emails for me today
    'massive_domain-test-002708',  # is there anything new in my mailbox
    'massive_domain-test-002799',  # do i have new emails
    'massive_domain-test-002818',  # please read new emails
    'massive_domain-test-002952',  # show me any emails received in the last hour
    'massive_domain-test-002957',  # read list of new emails
    'massive_domain-test-002947',  # any new emails received after four o'clock today
    'massive_domain-test-002825',  # show all emails sent to me from my boss
    'massive_domain-test-002920',  # please check my gmail for new mail
    'massive_domain-test-002740',  # what was the subject of the last email from mom
    'massive_domain-test-002962',  # please read my new mails
    'massive_domain-test-002625',  # do i have any notifications from social media
]
assign(D2, 'D2')

# -----------------------------------------------------------------------------
# D3  social-monitoring (scan social, see what's going on on platforms)
# -----------------------------------------------------------------------------
D3 = [
    'massive_domain-test-002633',  # who is doing facebook live right now
    'massive_domain-test-001185',  # friend updates
    'massive_domain-test-002617',  # please scan my social media and tell me what's happening
    'massive_domain-test-002646',  # tell me what's happening on instagram
    'massive_domain-test-001015',  # hot topic  -> ambiguous; could be social-trending; D3
    'massive_domain-test-001140',  # web searches  -> ambiguous; fragmentary -> H1
]
# Move 001140 to H1
D3 = [x for x in D3 if x != 'massive_domain-test-001140']
assign(D3, 'D3')

# -----------------------------------------------------------------------------
# E1  navigation-transit-booking (directions / transit / find shops / book ticket)
# -----------------------------------------------------------------------------
E1 = [
    'massive_domain-test-002044',  # find clothing stores within one mile
    'massive_domain-test-002207',  # book a ticket to ny by train
    'massive_domain-test-002167',  # find a train ticket to philadelphia
    'massive_domain-test-002199',  # find me a train ticket to boston
    'massive_domain-test-002172',  # lead the way to the place
    'massive_domain-test-002209',  # please book a ticket from jaipur to mumbai
    'massive_domain-test-002208',  # please book a ticket of rajdhani express train
    'massive_domain-test-002181',  # i need to get a ticket via train to orlando
    'massive_domain-test-002119',  # how do i get to the train station for a ticket
    'massive_domain-test-002161',  # how long will it take to get to the airport
    'massive_domain-test-002105',  # what shops are near me
    'massive_domain-test-002050',  # show me reviews of my nearest location food court
    'massive_domain-test-002079',  # is there a food festival in the area
    'massive_domain-test-002268',  # check movie theater prices for newly released movie in my location
    'massive_domain-test-002251',  # drive the car and bring my friends over  -> nav-y request -> E1
    'massive_domain-test-000419',  # call for take-out -> communication? It's calling restaurant -> E1 maybe; actually it's making a phone call to restaurant -> D1 outbound? It's local-business action. Place in E1 as "local services / takeout"
    'massive_domain-test-000029',  # can they provide takeaway  -> local-business inquiry -> E1
]
assign(E1, 'E1')

# -----------------------------------------------------------------------------
# E2  recipe-howto-instruction (how to cook / recipe / procedure)
# -----------------------------------------------------------------------------
E2 = [
    'massive_domain-test-001728',  # best oven temperature to roast potatoes
    'massive_domain-test-001731',  # easiest way to cook pasta
    'massive_domain-test-001718',  # tell me the recipe of
    'massive_domain-test-001754',  # find a recipe for dinner tonight
    'massive_domain-test-001729',  # best way to cook pasta al dente
    'massive_domain-test-001765',  # how long to cook a fifteen pound turkey
    'massive_domain-test-001734',  # list of ingredients for welsh rarebit
    'massive_domain-test-001732',  # ingredient instead of saffron
    'massive_domain-test-001780',  # find the recipe for sambar
]
assign(E2, 'E2')

# -----------------------------------------------------------------------------
# F1  smalltalk-personal-disclosure (chit-chat, jokes, complaints, self-statements)
# -----------------------------------------------------------------------------
F1 = [
    'massive_domain-test-000102',  # what's a good joke
    'massive_domain-test-000952',  # have i need to take a sun glass  -> ambiguous/self-question; F1 chit-chat
    'massive_domain-test-001139',  # what i like today  -> fragmentary chit-chat; F1
    'massive_domain-test-001084',  # i do not know how to answer this question you left a word
    'massive_domain-test-001093',  # i had food as soon as i got up
    'massive_domain-test-001105',  # my day was extremely hard
    'massive_domain-test-001096',  # met one of my old classmates today
    'massive_domain-test-001092',  # lets have a chat
    'massive_domain-test-001085',  # i am going to work today
    'massive_domain-test-001086',  # i had a bad day
    'massive_domain-test-000229',  # what's a good joke
    'massive_domain-test-000008',  # what's up olly
    'massive_domain-test-001168',  # what do you think about future
    'massive_domain-test-001276',  # a good first impression
    'massive_domain-test-001292',  # what's for today
    'massive_domain-test-000221',  # foreign policy of trump (treat as opinion-y trump query — already in A1 candidate; safer to keep as info-retrieval) -> A1
    'massive_domain-test-000002',  # pink is all we need
    'massive_domain-test-000995',  # opinion petabit -> fragmentary -> H1
    'massive_domain-test-002671',  # clear out the problem -> ambiguous fragmentary; H1
]
# Drop fragments + foreign-policy back to A1
F1.remove('massive_domain-test-000221')
assign(['massive_domain-test-000221'], 'A1')
F1.remove('massive_domain-test-000995')
F1.remove('massive_domain-test-002671')
assign(F1, 'F1')

# -----------------------------------------------------------------------------
# G1  shopping-list / to-do-list management (lists distinct from calendar/alarms)
# -----------------------------------------------------------------------------
G1 = [
    'massive_domain-test-001857',  # add red wine to my shopping list
    'massive_domain-test-001893',  # clear the list
    'massive_domain-test-001837',  # this week's shopping list
    'massive_domain-test-001936',  # take out the milk from the shopping list
    'massive_domain-test-001937',  # get rid of tax list from nineteen ninety
    'massive_domain-test-001858',  # remove the list called party time
    'massive_domain-test-001894',  # delete list for groceries
    'massive_domain-test-001849',  # reset my locations list
    'massive_domain-test-001946',  # please create new list
]
assign(G1, 'G1')

# -----------------------------------------------------------------------------
# H1  malformed / fragmentary / failed turn / unparseable
# -----------------------------------------------------------------------------
H1 = [
    'massive_domain-test-001126',  # technology
    'massive_domain-test-001003',  # a quarter to two
    'massive_domain-test-001015',  # already in D3 — remove from D3? Keep in D3
    'massive_domain-test-001140',  # web searches
    'massive_domain-test-000995',  # opinion petabit
    'massive_domain-test-002671',  # clear out the problem
]
# 001015 already assigned to D3; don't double-add
H1.remove('massive_domain-test-001015')
assign(H1, 'H1')

# -----------------------------------------------------------------------------
# Sanity check coverage
# -----------------------------------------------------------------------------
unassigned = [i for i in items if i not in assignments]
print(f"Unassigned: {len(unassigned)}")
for i in unassigned:
    print(f"  {i}: {items[i]}")

from collections import Counter
counts = Counter(assignments.values())
for k in sorted(counts):
    print(f"  {k}: {counts[k]}")
print(f"Total assigned: {sum(counts.values())} / {len(items)}")

# -----------------------------------------------------------------------------
# Write proposal
# -----------------------------------------------------------------------------
CLUSTER_DEFS = {
    'A1': dict(
        name='Factual information lookup',
        description=(
            "User asks the assistant a fact-based or knowledge question about the world or "
            "a person (history, geography, biography, general 'tell me about X')."
        ),
    ),
    'A2': dict(
        name='Status query on managed items',
        description=(
            "User asks about the current state of items the assistant manages — existing "
            "alarms, reminders, calendar events, lists, playlist contents, contact info."
        ),
    ),
    'A3': dict(
        name='Current-context lookup (weather/news/time/traffic/finance/scores)',
        description=(
            "User asks for current real-world information that changes over time: weather, "
            "news headlines, time, date, traffic, stock quotes, exchange rates, sports scores, "
            "what's playing in theaters."
        ),
    ),
    'A4': dict(
        name='Computation, conversion, and definition',
        description=(
            "Math expressions, unit/currency/time-zone conversions, and dictionary-style "
            "word definitions — short answers computable from a closed rule."
        ),
    ),
    'B1': dict(
        name='Smart-home / IoT actuation',
        description=(
            "User commands the assistant to actuate a physical device or setting in the "
            "home — turn on/off plugs and lights, change light color, start vacuum/coffee, "
            "adjust climate."
        ),
    ),
    'B2': dict(
        name='Media playback control (play/skip/resume)',
        description=(
            "User initiates or controls audio/video/game playback: play song/album/genre, "
            "play podcast/radio/audiobook, skip/replay/resume, start a game."
        ),
    ),
    'B3': dict(
        name='Media settings, volume, and now-playing info',
        description=(
            "User changes volume / playback mode / preferences during media use, rates or "
            "queues a song, or asks what's currently playing."
        ),
    ),
    'C1': dict(
        name='Schedule something (alarm / reminder / calendar event)',
        description=(
            "User asks the assistant to create a new alarm, reminder, or calendar event — "
            "any future-time scheduling action that the assistant must store."
        ),
    ),
    'C2': dict(
        name='Modify or remove existing schedule item',
        description=(
            "User removes, clears, or modifies an existing alarm, reminder, or calendar entry."
        ),
    ),
    'D1': dict(
        name='Outbound communication (send/post/reply)',
        description=(
            "User instructs the assistant to author and send a message on their behalf — "
            "email, tweet, SMS, social post, reply, complaint, customer-service contact."
        ),
    ),
    'D2': dict(
        name='Inbound communication check',
        description=(
            "User asks the assistant to check/read incoming email or message notifications, "
            "or query subjects/senders of recent inbound messages."
        ),
    ),
    'D3': dict(
        name='Social-feed monitoring',
        description=(
            "User asks the assistant to scan or summarize social-media activity — what "
            "friends/people are posting, trending topics, who is live, friend updates."
        ),
    ),
    'E1': dict(
        name='Navigation, transit, and local services',
        description=(
            "User asks for directions, travel time, nearby shops/restaurants/events, or "
            "books transit (train/airport) and local services (takeout)."
        ),
    ),
    'E2': dict(
        name='Recipes and how-to instructions',
        description=(
            "User requests cooking instructions, recipes, ingredient substitutions, or "
            "step-by-step procedures for a task."
        ),
    ),
    'F1': dict(
        name='Small-talk and personal disclosure',
        description=(
            "Chit-chat openers ('what's up'), jokes, opinions, and unprompted personal "
            "statements about how the user's day is going."
        ),
    ),
    'G1': dict(
        name='List and note management',
        description=(
            "Create, append to, remove from, or delete user-managed lists/notes "
            "(shopping list, to-do list, locations, named lists) — distinct from "
            "time-bound calendar/alarm items."
        ),
    ),
    'H1': dict(
        name='Fragmentary / unparseable utterance',
        description=(
            "Short single-word or unparseable utterances with no clear actionable intent — "
            "likely failed turn-taking or partial transcriptions."
        ),
    ),
}

clusters_out = []
for cid, defn in CLUSTER_DEFS.items():
    ids = sorted([i for i, c in assignments.items() if c == cid])
    if not ids:
        continue
    clusters_out.append({
        'name': defn['name'],
        'description': defn['description'],
        'text_ids': ids,
        'reasoning': (
            f"{len(ids)} of {len(items)} utterances ({100*len(ids)/len(items):.1f}%) "
            f"share the user-intent class '{defn['name'].lower()}'."
        ),
    })

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
suffix = uuid.uuid4().hex[:4]
out_path = WS / 'proposals' / f'prop_{ts}_{suffix}.json'

proposal = {
    'timestamp': datetime.now().isoformat(timespec='seconds'),
    'sample_size': len(items),
    'sample_strategy': 'shared_random_300',
    'style': 'user-intent-class lens (speech-act / goal)',
    'existing_clusters_considered': False,
    'clusters': clusters_out,
    'unclustered_ids': [],
    'observations': (
        "Lens: grouped utterances by the high-level *kind of interaction* the user is "
        "initiating (their goal/speech-act), not by the backend application. This produces "
        "16 clusters that cut across MASSIVE's surface domains. Notable consequences of "
        "this lens: (a) media-related utterances split into three intent classes — "
        "playback-control (play/skip), settings/volume/now-playing, and information-status "
        "queries about playlists — rather than collapsing into one 'music' bucket; "
        "(b) email/social splits into outbound-authoring, inbound-checking, and feed-"
        "monitoring, separating composition from consumption; (c) calendar/alarms split "
        "into create vs. modify/delete, since the speech-act differs even though the "
        "application is the same; (d) lists/notes are pulled out of scheduling because "
        "they are timeless containers rather than time-bound actions; (e) factual lookup "
        "splits by whether the answer is static-world-knowledge (A1), current-context "
        "(A3), or computable/closed-form (A4). A small fragmentary/unparseable bucket "
        "(H1, n=5) holds utterances like 'technology', 'web searches', and 'a quarter to "
        "two' that have no recoverable user goal."
    ),
}

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(proposal, f, indent=2, ensure_ascii=False)

print(f"Wrote proposal: {out_path}")
print(f"k = {len(clusters_out)}")
