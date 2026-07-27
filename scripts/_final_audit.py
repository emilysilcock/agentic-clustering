"""Build the final audit JSON for the massive_intent workspace.

Assignments encoded as (index, cluster_id, confidence, optional note).
Index refers to position in the 300-sample list pulled from sample.json.
"""

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

SAMPLE = Path(r"C:\Users\emily\.claude\projects\C--Users-emily-Documents-agentic-clustering\42c9eba3-220a-4d91-b8f1-c29d2fe9a2d3\tool-results\bl1vjnldl.txt")
WORKSPACE = Path(r"C:\Users\emily\Documents\agentic-clustering\results\clustering\massive_intent\seed=0_proposers_v1")

# Each entry: (cluster_id_or_None, confidence_1_5, optional_note)
# Indices match the 300-sample order.
ASSIGNMENTS = [
    # 0  what is the time difference between here and japan
    ("c28", 5, None),
    # 1  turn off the shed light
    ("c39", 5, None),
    # 2  i wish it could give me advice  (chat-share-mood-or-event style)
    ("c55", 3, "open-ended wish; chat-y, not a concrete action"),
    # 3  text an email to donna hey what are you doing today
    ("c17", 4, "phrased 'text an email' but body provided => compose+send"),
    # 4  math (bare noun)
    ("c57", 4, None),
    # 5  clean my house
    ("c40", 4, "implicit vacuum/appliance trigger"),
    # 6  olly tweet apple that the iphone doesn't work  (complaint to company)
    ("c22", 5, None),
    # 7  remove my task list
    ("c52", 5, None),
    # 8  tell me when i should leave for a scheduled event so that i am on time
    ("c7", 4, "conditional pre-event alert => reminder-create"),
    # 9  erase that from my calendar
    ("c6", 5, None),
    # 10 show me the current time in germany
    ("c28", 5, None),
    # 11 please check my emails for me
    ("c19", 4, "ambiguous between check-new and read; closer to check"),
    # 12 please lower music volume
    ("c42", 5, None),
    # 13 i want to remove apples from list
    ("c52", 5, None),
    # 14 disable alarm for three p. m.
    ("c3", 5, None),
    # 15 play my shuffled playlist please
    ("c11", 5, None),
    # 16 turn off sound
    ("c42", 5, "mute the audio"),
    # 17 please set a notification on twenty third october about meeting with my colleague
    ("c7", 4, "could be calendar event or reminder; phrased as 'notification'"),
    # 18 silence for two hours
    ("c42", 5, "timed quiet mode"),
    # 19 give me an update on the election in one hour
    ("c7", 3, "timed news alert -> reminder; could also be news"),
    # 20 turn the light off in the living room
    ("c39", 5, None),
    # 21 show me the train schedules to the metropolitan opera house
    ("c44", 5, None),
    # 22 remove tonights dinner with shelly
    ("c6", 5, None),
    # 23 what is the weather
    ("c25", 5, None),
    # 24 play one more time
    ("c12", 4, "replay current track"),
    # 25 is there anything important happening on social media
    ("c23", 5, None),
    # 26 let's play
    ("c57", 3, "underspecified; could be music/game"),
    # 27 can you tell me the contact information of jo
    ("c49", 5, None),
    # 28 what is two hundred divided by ten
    ("c36", 5, None),
    # 29 make a new shopping list
    ("c50", 5, None),
    # 30 put on  (very underspecified)
    ("c57", 3, None),
    # 31 wake me up at five am
    ("c1", 5, None),
    # 32 what do i have to do this week
    ("c8", 5, None),
    # 33 how many lists do i have in contacts
    ("c49", 3, "ambiguous: contact-list count; closest to contacts"),
    # 34 i need good ideas for cooking
    ("c37", 4, None),
    # 35 how long will it take to travel to japan from south korea
    ("c45", 4, "travel time; closer to navigation/directions"),
    # 36 where is the pharmacy in leavenworth
    ("c46", 5, None),
    # 37 wake me at six am thursday so i have time for the meeting
    ("c1", 5, None),
    # 38 tweet samsung and tell them to stop killing people with their products
    ("c22", 5, None),
    # 39 how many of my contacts live or work in detroit
    ("c49", 5, None),
    # 40 call taxi
    ("c43", 4, "book transport"),
    # 41 tell interesting news
    ("c27", 5, None),
    # 42 how's the weather like in new jersey
    ("c25", 5, None),
    # 43 what should i make for dinner
    ("c37", 5, None),
    # 44 list me the stock of apple right now
    ("c34", 5, None),
    # 45 lower the lights please
    ("c38", 4, "brightness down"),
    # 46 can you tell me the time it is
    ("c28", 5, None),
    # 47 don't play this song again
    ("c13", 4, "block/dislike this song"),
    # 48 notify me when joshua emails me
    ("c7", 5, "email-arrival reminder per c7 description"),
    # 49 what's on your mind
    ("c54", 4, "small talk opener"),
    # 50 how many euros can i get for one dollar
    ("c35", 5, None),
    # 51 get hourly notification on sports news
    ("c7", 4, "recurring news alert"),
    # 52 please show me the list that i have
    ("c8", 3, "could be list or todo; reads as list display"),
    # 53 play only my list
    ("c11", 5, None),
    # 54 check most current emails
    ("c19", 3, "ambiguous between check-new and read"),
    # 55 add conference call at four p. m. to my reminders for today
    ("c7", 5, None),
    # 56 has jane doe emailed me
    ("c19", 5, None),
    # 57 could you please email john saying i'm on leave
    ("c17", 5, None),
    # 58 show me my latest social media activity
    ("c23", 5, None),
    # 59 what reminders do i still have
    ("c8", 5, None),
    # 60 change the lights to dim
    ("c38", 5, None),
    # 61 which contact haven't i called in twelve months
    ("c49", 4, None),
    # 62 can you remind me to order the turkey three weeks before thanksgiving
    ("c7", 5, None),
    # 63 please check my emails and notify me if exists any new email
    ("c19", 5, None),
    # 64 play love songs
    ("c10", 5, "by mood/genre"),
    # 65 show me the best podcast of rock songs having good rating
    ("c16", 4, "podcast surface"),
    # 66 i want to know what supermarket near me has the best price on gluten free bread
    ("c46", 5, None),
    # 67 instructions to make a meal
    ("c37", 5, None),
    # 68 can you open my itunes
    ("c53", 5, None),
    # 69 please play a song by bruno mars
    ("c9", 5, None),
    # 70 olly i feel like dancing play me some rock north roll
    ("c10", 5, None),
    # 71 do you have the train times for beaumont tx
    ("c44", 5, None),
    # 72 olly do i have any new emails
    ("c19", 5, None),
    # 73 enable my plug
    ("c39", 5, None),
    # 74 play game
    ("c53", 5, None),
    # 75 fastest route to turiellos
    ("c45", 5, None),
    # 76 tell me about the cheapest flight fares to nj today
    ("c43", 4, "ticket-pricing query"),
    # 77 set notifications on the current weather disasters in america
    ("c7", 4, "weather-alert reminder"),
    # 78 how is lasagne made
    ("c37", 5, None),
    # 79 send me an alert and hour before my next appointment
    ("c7", 5, "meeting pre-alert per c7"),
    # 80 hello how is your day
    ("c54", 5, None),
    # 81 send email to boss saying i will be late
    ("c17", 5, None),
    # 82 let's have the lights blue
    ("c38", 5, None),
    # 83 cancel all my appointments
    ("c6", 5, None),
    # 84 i need a good cry can you play me sad rock songs
    ("c10", 5, None),
    # 85 trash my list
    ("c52", 5, None),
    # 86 add broccoli to my grocery list
    ("c51", 5, None),
    # 87 please check the trending topics on twitter
    ("c23", 5, None),
    # 88 read some more of the daisy goodwin book for me
    ("c16", 5, "audiobook resume"),
    # 89 crack a joke
    ("c56", 5, None),
    # 90 remind me at  (incomplete utterance)
    ("c7", 2, "fragment; intent is reminder-create but content missing"),
    # 91 what's shakin bacon
    ("c54", 4, "greeting/small talk"),
    # 92 consequences to actions  (bare phrase)
    ("c57", 4, None),
    # 93 has anyone commented on my status
    ("c23", 5, None),
    # 94 set an alarm for tomorrow at six in the morning
    ("c1", 5, None),
    # 95 i need to get up at ten tomorrow
    ("c1", 5, None),
    # 96 it's dirty here make some noise  (very odd; closest: appliance/vacuum)
    ("c40", 2, "ambiguous; reads as vacuum cue but oddly phrased"),
    # 97 what is the latest story from fox news
    ("c27", 5, None),
    # 98 i need a ride home
    ("c43", 4, "transport book/request"),
    # 99 alexa please turn down the lights in the house
    ("c38", 5, None),
    # 100 what events are happening soon
    ("c46", 3, "local events ambiguous; closer to local discovery"),
    # 101 give me the description of a television circuit
    ("c31", 4, "description/definition"),
    # 102 what's the temperature outside
    ("c25", 5, None),
    # 103 play techno music
    ("c10", 5, None),
    # 104 i want you to play the podcast
    ("c16", 5, None),
    # 105 quiet mode on until i am home from work
    ("c42", 5, None),
    # 106 give the list of theaters in the vicinity
    ("c46", 4, "nearby venues; movie-showtimes also plausible"),
    # 107 how tall is brad pitt
    ("c32", 5, None),
    # 108 add practice to calendar on feb four at king's park at two p. m.
    ("c4", 5, None),
    # 109 my vacuum cleaner should start between ten to eleven am everyday
    ("c40", 5, None),
    # 110 can you reply to charlotte that i am going to be busy with the projects for next three weeks
    ("c18", 4, "could be reply or message; 'reply to charlotte' => email-reply"),
    # 111 play the next episode of a podcast
    ("c16", 5, None),
    # 112 nobody knows  (fragment)
    ("c57", 3, None),
    # 113 tell me where steve jobs was born
    ("c32", 5, None),
    # 114 give me the list of available train tickets from edinburgh to leeds
    ("c43", 5, None),
    # 115 where's the closest zoo to where i'm at
    ("c46", 4, "local places query"),
    # 116 i want an alarm for three today
    ("c1", 5, None),
    # 117 display the seven day forecast for this week
    ("c25", 5, None),
    # 118 tell me about alarms
    ("c2", 5, None),
    # 119 where is the nearest walmart
    ("c46", 5, None),
    # 120 add meeting reminder for sunday with my parents in law
    ("c7", 4, "phrased as reminder; could be calendar"),
    # 121 my day was great
    ("c55", 5, None),
    # 122 what is the forecast for the week
    ("c25", 5, None),
    # 123 stop audiobook
    ("c12", 4, "playback control across media; closest is c12"),
    # 124 can you describe a credit card
    ("c31", 4, None),
    # 125 play coldplay album
    ("c9", 5, None),
    # 126 ok google where does sophia vergara live
    ("c32", 5, None),
    # 127 power off please
    ("c39", 3, "ambiguous: device off or app close"),
    # 128 what are the b. b. c. poll predictions for the upcoming us elections
    ("c27", 4, "news source query"),
    # 129 set an alarm for six thirty am
    ("c1", 5, None),
    # 130 email alice to let her know we are on the way
    ("c17", 5, None),
    # 131 what ingredients do i need to bake a large cake
    ("c37", 5, None),
    # 132 add this new email with contact
    ("c49", 3, "ambiguous; reads as add contact"),
    # 133 remember this email
    ("c20", 2, "could be save/flag email; not a clean fit"),
    # 134 coordinate all pop song genres
    ("c10", 3, "ambiguous music command, by genre"),
    # 135 can you contact samsung and say my washer is on fire
    ("c22", 4, "complaint to company"),
    # 136 book a taxi to go to the movies at one
    ("c43", 5, None),
    # 137 how big is the empire state building
    ("c30", 5, None),
    # 138 will you mark that i am busy for an event tomorrow at two p. m.
    ("c4", 4, "calendar event create with busy status"),
    # 139 i would like to reply to the most recent email
    ("c18", 5, None),
    # 140 tell me everything you know about sloths
    ("c30", 5, None),
    # 141 how much is one american dollars worth in england
    ("c35", 5, None),
    # 142 what are some theme parks nearby
    ("c46", 5, None),
    # 143 where is the car
    ("c30", 2, "fragment; could be navigation/factoid"),
    # 144 what jacket should i wear
    ("c26", 5, None),
    # 145 how tall is hulk hogan
    ("c32", 5, None),
    # 146 send email to joseph at gmail dot com
    ("c17", 5, None),
    # 147 train times from near me to location
    ("c44", 5, None),
    # 148 how to get somewhere  (fragment about navigation)
    ("c45", 3, "underspecified directions"),
    # 149 how many people are attending my next meeting
    ("c5", 5, None),
    # 150 what alarms i have set
    ("c2", 5, None),
    # 151 tell me when it is five p. m.
    ("c7", 4, "trigger-based reminder"),
    # 152 wooo  (exclamation)
    (None, 1, "no actionable intent"),
    # 153 check for emails from steve
    ("c20", 4, "filtered email read"),
    # 154 please turn up the screen brightness all the way
    ("c38", 4, "screen brightness fits lights-color-or-brightness loosely"),
    # 155 go to the washington post website
    ("c53", 3, "open app/site"),
    # 156 olly tell me a joke
    ("c56", 5, None),
    # 157 cisco system  (bare noun)
    ("c57", 4, None),
    # 158 which ocean touches at our continent
    ("c30", 5, None),
    # 159 is it ten  (asking time)
    ("c28", 4, None),
    # 160 play biography of jackie kennedy
    ("c16", 4, "audiobook-style biography"),
    # 161 change the lights to a different hue
    ("c38", 5, None),
    # 162 is it anyone i knows birthday this month
    ("c5", 3, "calendar query about contacts' birthdays"),
    # 163 play my radio station
    ("c15", 5, None),
    # 164 play the oldies station
    ("c15", 5, None),
    # 165 what color is a dragon fruit
    ("c30", 5, None),
    # 166 delete events from calendar
    ("c6", 5, None),
    # 167 what movie can i watch tonight on the theater here in boston
    ("c47", 5, None),
    # 168 erase all  (fragment)
    ("c52", 3, "list/wipe ambiguous"),
    # 169 list of famous biryani recipes
    ("c37", 5, None),
    # 170 what is the population of new york
    ("c30", 5, None),
    # 171 anything interesting from bob's news
    ("c27", 5, None),
    # 172 olly turn the lights off in the bedroom
    ("c39", 5, "could also be c38; bedroom lights, off is binary"),
    # 173 i want to listen radio
    ("c15", 5, None),
    # 174 tell me about the latest political news
    ("c27", 5, None),
    # 175 i hate this song
    ("c13", 5, None),
    # 176 do you know math
    ("c54", 3, "small talk / capability question"),
    # 177 turn on favorite songs
    ("c11", 4, "play personal playlist"),
    # 178 what's jlo up to
    ("c32", 5, None),
    # 179 away off from list
    ("c52", 3, "garbled list removal"),
    # 180 go on sweet talk me
    ("c54", 4, "open chat invitation"),
    # 181 respond to my bosses email with the word that i will be in at four in the evening
    ("c18", 5, None),
    # 182 open the radio app
    ("c53", 4, "could be radio-play or app launch"),
    # 183 robot vacuum the living room now
    ("c40", 5, None),
    # 184 what lists do i have queried
    ("c8", 3, "ambiguous lists query"),
    # 185 can you delete the dentist appointment
    ("c6", 5, None),
    # 186 send email to jessica
    ("c17", 5, None),
    # 187 the song in background is cool
    ("c13", 4, "positive feedback on current song"),
    # 188 what is the current weather condition in my place
    ("c25", 5, None),
    # 189 can you increase the brightness in the room
    ("c38", 5, None),
    # 190 what are lady gagas most popular songs
    ("c32", 3, "person-info or factoid; reads as factoid about Lady Gaga's songs"),
    # 191 please summarize the latest george r. r. martin ice and fire book
    ("c30", 3, "book summary; closest factoid"),
    # 192 add tal meeting on twenty first at seven
    ("c4", 5, None),
    # 193 i want to order chinese take-out
    ("c48", 5, None),
    # 194 please show me any articles related to weather in the morning news
    ("c27", 5, None),
    # 195 get rid of all my scheduled events
    ("c6", 5, None),
    # 196 what else is missing in the diary
    ("c5", 3, "calendar/diary query"),
    # 197 what is a hypothesis
    ("c31", 5, None),
    # 198 play chess
    ("c53", 5, None),
    # 199 cut down the volume
    ("c42", 5, None),
    # 200 boiling  (bare noun)
    ("c57", 4, None),
    # 201 play my favorites
    ("c11", 5, None),
    # 202 play the next one
    ("c12", 5, None),
    # 203 add my opinion to this song great
    ("c13", 4, None),
    # 204 play the most recent podcast for this american life
    ("c16", 5, None),
    # 205 are there any meetings set for next wednesday
    ("c5", 5, None),
    # 206 how is the weather where i am
    ("c25", 5, None),
    # 207 run the vacuum
    ("c40", 5, None),
    # 208 tell me the time in this time zone
    ("c28", 5, None),
    # 209 can you play rock music for the next hour
    ("c10", 5, None),
    # 210 show bio of rihana
    ("c32", 5, None),
    # 211 thirteenth june is a day of election result please set it
    ("c4", 4, "calendar event add"),
    # 212 what is the timing of bagmati express
    ("c44", 5, None),
    # 213 reduce volume
    ("c42", 5, None),
    # 214 please define forensic
    ("c31", 5, None),
    # 215 what events are going in my town this week
    ("c46", 4, "local events"),
    # 216 olly what day of the week is halloween
    ("c29", 5, None),
    # 217 what do i have coming up
    ("c5", 4, "ambiguous between calendar query and reminders"),
    # 218 who sings the song about a long black train
    ("c30", 4, "factoid music question"),
    # 219 tell me the local events
    ("c46", 5, None),
    # 220 the second of next mont lands on what day
    ("c29", 5, None),
    # 221 remind me about my schedule for the afternoon
    ("c5", 3, "ambiguous: calendar-query vs reminder-query"),
    # 222 give me the list of circus shows going on in the city right now
    ("c46", 4, "local events listing"),
    # 223 clear out the shopping list
    ("c52", 5, None),
    # 224 tell me if global warming is true
    ("c30", 4, None),
    # 225 play audiobook of planets
    ("c16", 5, None),
    # 226 i will require full cover jacket if it is too stormy in evening
    ("c26", 3, "weather-derived statement; not a clean ask"),
    # 227 what's the name of the piece you are playing
    ("c14", 5, None),
    # 228 i need to find a gift what stores are within a one mile radius
    ("c46", 5, None),
    # 229 make the coffee
    ("c40", 5, None),
    # 230 remind me tomorrow at ten am about the meeting
    ("c7", 5, None),
    # 231 what's the latest news
    ("c27", 5, None),
    # 232 play for me the music by the beatles
    ("c9", 5, None),
    # 233 is it going to rain at one p. m. today
    ("c25", 5, None),
    # 234 i want tickets to the sold out concert on saturday night
    ("c43", 3, "event ticket; closest existing cluster is transport tickets"),
    # 235 hoover the hallway
    ("c40", 5, None),
    # 236 is this program is scheduled
    ("c5", 3, "garbled calendar-style query"),
    # 237 what day is the fifth
    ("c29", 5, None),
    # 238 how many eggs do i need for an omelet
    ("c37", 5, None),
    # 239 play slayer
    ("c9", 5, None),
    # 240 veganism  (bare noun)
    ("c57", 4, None),
    # 241 what is the definition of logic
    ("c31", 5, None),
    # 242 remind me tonight to pick up my dry cleaning at eight p. m.
    ("c7", 5, None),
    # 243 open tasks delete future events
    ("c6", 4, None),
    # 244 how would you describe a ball
    ("c31", 4, None),
    # 245 how do i reach sarah
    ("c49", 5, None),
    # 246 check mom's number
    ("c49", 5, None),
    # 247 what is the largest active volcano on earth
    ("c30", 5, None),
    # 248 set my calendar to remind me to buy groceries every friday
    ("c7", 4, "recurring reminder; phrased as 'calendar' but reminder-shaped"),
    # 249 put the lights off now
    ("c39", 5, None),
    # 250 set the alarm off
    ("c3", 4, "remove/disable alarm"),
    # 251 could you tell me the time in london
    ("c28", 5, None),
    # 252 play next
    ("c12", 5, None),
    # 253 turn the lights to red color
    ("c38", 5, None),
    # 254 are you smart
    ("c54", 4, "small talk capability check"),
    # 255 define speaker
    ("c31", 5, None),
    # 256 please add list of things to buy for party
    ("c50", 5, None),
    # 257 play my favorite book
    ("c16", 5, None),
    # 258 what is four plus five
    ("c36", 5, None),
    # 259 will the temperature be higher than forty tomorrow
    ("c25", 5, None),
    # 260 mute for fifteen minutes
    ("c42", 5, None),
    # 261 delete all the events in my calendar
    ("c6", 5, None),
    # 262 are there any new emails in outlook
    ("c19", 5, None),
    # 263 event reminder mona tuesday
    ("c7", 4, None),
    # 264 android  (bare noun)
    ("c57", 4, None),
    # 265 please clean my shopping list
    ("c52", 5, None),
    # 266 radio please
    ("c15", 5, None),
    # 267 tell me what the weather is doing in grand rapids mi right now
    ("c25", 5, None),
    # 268 tweet my current location
    ("c21", 5, None),
    # 269 define elaborate
    ("c31", 5, None),
    # 270 cancel everything on my calendar
    ("c6", 5, None),
    # 271 find information on today's stocks
    ("c34", 5, None),
    # 272 delete calendar item
    ("c6", 5, None),
    # 273 can you tell me the times the train leaves for chicago
    ("c44", 5, None),
    # 274 i need to get a ticket via train to orlando from hwood
    ("c43", 5, None),
    # 275 what is the size of the united states
    ("c30", 5, None),
    # 276 what is the time in las vegas
    ("c28", 5, None),
    # 277 raise the volume of the current music
    ("c42", 5, None),
    # 278 play for me music by the beatles
    ("c9", 5, None),
    # 279 what's the recipe for fish soup
    ("c37", 5, None),
    # 280 what is robin williams birthday
    ("c32", 5, None),
    # 281 save this song to playlist
    ("c51", 5, None),
    # 282 do i have anything planned for the twenty first
    ("c5", 5, None),
    # 283 what is the time difference between california and new york
    ("c28", 5, None),
    # 284 tell me how to make a poor boy sandwich
    ("c37", 5, None),
    # 285 play that podcast i was listening to yesterday
    ("c16", 5, None),
    # 286 do i need a sweater today
    ("c26", 5, None),
    # 287 fast  (fragment)
    ("c57", 3, None),
    # 288 play and shuffle all slow music songs
    ("c10", 5, None),
    # 289 what did i tell susan in my last email
    ("c20", 5, None),
    # 290 when is my brunch with jennifer
    ("c5", 5, None),
    # 291 tell me the info about india's geography
    ("c30", 5, None),
    # 292 i want to hear any songs that got grammys this year
    ("c10", 4, "songs by criterion; closest is genre/mood"),
    # 293 play me a game of tic tac toe
    ("c53", 5, None),
    # 294 tweet to apple about non receipt of iphone i sent for repairs last week
    ("c22", 5, None),
    # 295 start radio channel eight hundred and eighty nine
    ("c15", 5, None),
    # 296 what do i need to make lamb pathia
    ("c37", 5, None),
    # 297 describe a sloth
    ("c31", 4, None),
    # 298 give me news from c. n. n.
    ("c27", 5, None),
    # 299 what's the next event at the library
    ("c46", 4, "local event"),
]


def main():
    sample = json.load(open(SAMPLE, encoding="utf-8"))
    assert len(sample) == len(ASSIGNMENTS), f"sample={len(sample)} assignments={len(ASSIGNMENTS)}"

    # Read cluster_version from state.json
    state = json.load(open(WORKSPACE / "state.json", encoding="utf-8"))
    cluster_version = state["meta"]["cluster_version"]
    print(f"cluster_version from state.json: {cluster_version}")

    assignments = []
    for text, (cid, conf, note) in zip(sample, ASSIGNMENTS):
        entry = {
            "text_id": text["id"],
            "cluster_id": cid,
            "confidence": conf,
        }
        if note:
            entry["note"] = note
        assignments.append(entry)

    # quick coverage / mean confidence preview
    n = len(assignments)
    nulls = sum(1 for a in assignments if a["cluster_id"] is None)
    coverage = (n - nulls) / n
    mean_conf = sum(a["confidence"] for a in assignments) / n
    print(f"n={n} coverage={coverage:.4f} mean_conf={mean_conf:.3f}")

    # weak observations: clusters whose mean confidence on this audit is low
    from collections import defaultdict
    per_cluster = defaultdict(list)
    for a in assignments:
        if a["cluster_id"]:
            per_cluster[a["cluster_id"]].append(a["confidence"])
    weak = []
    for cid, confs in per_cluster.items():
        if len(confs) >= 3 and sum(confs) / len(confs) < 3.5:
            weak.append({"cluster_id": cid, "mean_confidence": round(sum(confs)/len(confs), 2), "n": len(confs)})

    audit_doc = {
        "audit_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cluster_definitions_version": cluster_version,
        "sample_size": n,
        "sample_method": "random, exclude-seen",
        "assignments": assignments,
        "summary": {
            "weak_clusters": weak,
            "observations": [
                "Description rewrites resolved the previously-flagged weak spots: c12 music-playback-control now cleanly absorbs 'play next' / 'play one more time' / 'stop audiobook'; c27 news-query carved out local events to c46; c39/c40 binary-vs-cycle split holds for plug/light vs vacuum/coffee/oven; c42 volume-or-mute now reliably catches 'silence for X', 'quiet mode', 'turn off sound'.",
                "Lowest-confidence residual is c57 fragmentary-or-topic-query (mean ~3.6) and c55 chat-share-mood-or-event — both are inherently noisy intent-buckets, not description bugs.",
                "Recurring low-confidence ambiguities are cross-cluster boundaries: reminder-create (c7) vs calendar-event-create (c4) for 'set notification on date X about Y'; email-check-new (c19) vs email-read-content (c20) for 'please check my emails'; reminder-query (c8) vs calendar-event-query (c5) for 'remind me about my schedule'.",
                "A handful of fragments ('wooo', 'remind me at', 'remember this email', 'where is the car') don't fit any cluster cleanly; assigned at confidence 2 or null where no fit existed.",
            ],
        },
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit_id_short = audit_doc["audit_id"][:8]
    out_path = WORKSPACE / "audits" / f"audit_{ts}_{audit_id_short}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit_doc, f, indent=2, ensure_ascii=False)
    print(f"wrote {out_path}")
    print(out_path)


if __name__ == "__main__":
    main()
