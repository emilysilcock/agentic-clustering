"""Build proposer_b (valence-intensity) proposal for the goemotions sample.

One-off script — emits the JSON proposal to the workspace's proposals/ dir.
The assignment dict below was produced by reading all 300 sampled texts and
classifying each by valence (pos/neg/mixed) x intensity (mild/moderate/strong)
x affect type (per user instructions: primary emotion). Ambiguous / neutral
/ purely factual texts go in unclustered_ids per task spec.
"""
import json, datetime, uuid, os
from pathlib import Path

SAMPLE_PATH = Path(r"C:/Users/emily/AppData/Local/Temp/proposer_b_sample.json")
WORKSPACE = Path(r"C:/Users/emily/Documents/agentic-clustering/results/clustering/goemotions/seed=0_proposers_v0")

sample = json.load(open(SAMPLE_PATH, encoding="utf-8"))
id2text = {x["id"]: x["text"] for x in sample}
all_ids = set(id2text)

# Each cluster: name, description, list of text_ids, reasoning sketch.
clusters = [
    # ============ POSITIVE / STRONG ============
    {
        "name": "Strong admiration & love",
        "description": "Intense positive appraisal of a person, work, or object — superlative praise, declarations of love, deep enthusiasm.",
        "text_ids": [
            "goemotions-all-022984",  # "I can't tell you how much I appreciate it. <3 Thank you."
            "goemotions-all-021491",  # "I love me some quality evil dead"
            "goemotions-all-041326",  # "Love him."
            "goemotions-all-018394",  # "I LOVE this."
            "goemotions-all-044618",  # "You. I like you."
            "goemotions-all-006728",  # "Truly a beautiful story, 11/10 :')"
            "goemotions-all-046523",  # "It is so fucking beautiful"
            "goemotions-all-046968",  # wholesome encounters keep my hopes up
            "goemotions-all-018791",  # "[NAME] and [NAME] were awesome."
            "goemotions-all-051378",  # soundtrack one of the best parts of the game. it's incredible
            "goemotions-all-049598",  # "Da real MVP"
            "goemotions-all-012398",  # "A cat has never understood me so well"
            "goemotions-all-012149",  # "I love me some RFM"
            "goemotions-all-045787",  # "this is my favorite scene from girl with the dragon tattoo"
        ],
        "reasoning": "14 texts express strong, unqualified admiration or love. Often use 'love', 'beautiful', superlatives, or hearts."
    },
    {
        "name": "Intense joy / excitement",
        "description": "High-arousal positive — exclamation, shouting, celebratory enthusiasm.",
        "text_ids": [
            "goemotions-all-015109",  # "Yes!!!!"
            "goemotions-all-033728",  # "This would be huge!"
            "goemotions-all-003867",  # "It's showtime!!"
            "goemotions-all-028841",  # ending the quarter w/ some point [NAME] action, I love it
            "goemotions-all-052267",  # "I'm literally screaming."
            "goemotions-all-052104",  # Keep going! Every day you stick with it is another victory you can cherish!
        ],
        "reasoning": "6 texts with high-arousal positive markers (multi-bang, all-caps screaming, hype phrases)."
    },
    {
        "name": "Deep gratitude",
        "description": "Heartfelt thanks beyond routine politeness — extended, emphasized, or punctuated with affection.",
        "text_ids": [
            "goemotions-all-050770",  # thank you SO SO much for your help <3
            "goemotions-all-018600",  # "Appreciate you!"
            "goemotions-all-039133",  # "Lol, thank you for my first upvote and for contacting the mods."
            "goemotions-all-027501",  # "Thank you [NAME] master"
            "goemotions-all-039133",  # "Lol, thank you for my first upvote and for contacting the mods."
        ],
        "reasoning": "5 texts go beyond a flat 'thanks' — emphasis (SO SO), hearts, or addressing the helper warmly."
    },

    # ============ POSITIVE / MODERATE ============
    {
        "name": "Approval & agreement",
        "description": "Endorsing another comment, confirming correctness, or agreeing — positive but measured.",
        "text_ids": [
            "goemotions-all-032820",  # Don't worry, I'm pretty sure you're right and he's wrong
            "goemotions-all-008698",  # "he's obviously helping"
            "goemotions-all-038074",  # I was jumping the same gun but decided to do some googling :)
            "goemotions-all-002495",  # "Well I am happy you believe that."
            "goemotions-all-036086",  # "it's great"
            "goemotions-all-019499",  # "True....I know!"
            "goemotions-all-041698",  # "This is legit. Nobody puts baby in a corner!"
            "goemotions-all-015456",  # "Be ready to Experience Canes Hockey"
            "goemotions-all-015886",  # "Exact same here. I blame the egg nog!"
            "goemotions-all-015056",  # "That's awesome! Good call :)"
            "goemotions-all-009342",  # "Amen! However, see my post about the problem with drafting [NAME]"
            "goemotions-all-023454",  # "No problem here. Bio hardly ever matches the personality"
        ],
        "reasoning": "Affirming-with-substance: 'you're right', 'true', 'exact same here'."
    },
    {
        "name": "Amusement / humor appreciation",
        "description": "Finding something funny — laughing, joking, sharing a pun or absurd remark with positive tone.",
        "text_ids": [
            "goemotions-all-008847",  # "Wow spongebob was right!"
            "goemotions-all-037256",  # "I did nazi that coming!" (pun)
            "goemotions-all-029667",  # "Ikr. Who the hell goes to Greggs for *pizza*?"
            "goemotions-all-029532",  # "They all clapped."
            "goemotions-all-018316",  # "Black goodness 😂😂"
            "goemotions-all-018197",  # "Thermal shock has GOT to be some indie bands name"
            "goemotions-all-009182",  # "I chuckled, sue me"
            "goemotions-all-054082",  # "I always laugh at the statement above..."
            "goemotions-all-024996",  # "lol @ the dude lying down in the hut at 2:12"
            "goemotions-all-001444",  # "the dog is sneezing which is canine for 'this is a great game'"
            "goemotions-all-038945",  # "This comment killed me. RIP in pieces."
            "goemotions-all-050230",  # "I'll take PVZ. Your flair: 'Maybe necks time'"
            "goemotions-all-046886",  # "...... mmmm unforbidden ham"
            "goemotions-all-035641",  # "Or... Or you can eat mine."
            "goemotions-all-037821",  # "school announcement: Remember not to commit thought crimes..."
            "goemotions-all-048258",  # "Fun fact: the average person eats 5 dogs in their sleep, checkmate veegunz"
            "goemotions-all-019919",  # "OP may have been avoiding [NAME] jokes."
            "goemotions-all-018343",  # "oh the irony of asking for nudes on a device that can conjure all porn..."
            "goemotions-all-018636",  # "He didn't see him and afterwards tried to high five him but [NAME] was walking away"
            "goemotions-all-002742",  # "> they hardly ever land shots on you. Welcome to console shooters."
            "goemotions-all-045227",  # "That was a joke omg"
            "goemotions-all-014216",  # "It's about 20 years. They let me out early because of good behavior."
            "goemotions-all-011427",  # "That's ma'am!"
            "goemotions-all-023558",  # "He can hook him up with a pink blouse. Courtesy of big mamma."
            "goemotions-all-039777",  # "No,no wheel chairs, freedom alas."
            "goemotions-all-043143",  # "The Dark Knight that didn't rise?? Home Alone.. forever?"
            "goemotions-all-051337",  # "T H I C C D O N is actually zaddy af."
            "goemotions-all-054127",  # "Just don't cross the Tigers, hmmmm?"
            "goemotions-all-036864",  # "Holy shit they fornicate like rabbits!!"
        ],
        "reasoning": "Laughter markers (lol, 😂, 'I chuckled', 'killed me'), puns, absurdist quips and one-liners delivered for laughs."
    },
    {
        "name": "Fondness & affection (warmth)",
        "description": "Gentle warm feeling toward a person, pet, or memory — softer than 'love' but clearly positive bond.",
        "text_ids": [
            "goemotions-all-014594",  # "Snuggling with my boyfriend"
            "goemotions-all-006340",  # "Right on. Hi from saskatoon canada. Take care buddy"
            "goemotions-all-013880",  # "I feel a connection to this woman"
            "goemotions-all-034909",  # "Sounds sweet man, nice memory to have!"
            "goemotions-all-024691",  # "hope they had fun in the snow!"
            "goemotions-all-030568",  # "Baby's feeling their look too!"
            "goemotions-all-043540",  # "Rover has treated us well."
            "goemotions-all-024818",  # "I won't say no to a new group!"
        ],
        "reasoning": "Affectionate but mild — wishing well, fondly remembering, cozy intimacy."
    },

    # ============ POSITIVE / MILD ============
    {
        "name": "Mild approval / casual acknowledgement",
        "description": "Brief polite positive — 'right on', 'cool', 'nice', confirming without strong feeling.",
        "text_ids": [
            "goemotions-all-034217",  # "America is cool"
            "goemotions-all-027239",  # "yea, get back to me when the deal is done"
            "goemotions-all-041866",  # "A surprise to be sure, but a welcome one."
            "goemotions-all-027919",  # status update about playing
            "goemotions-all-045976",  # "That Pixel art yak is so cute tho."
            "goemotions-all-025028",  # "Read the poem... I liked it."
            "goemotions-all-049611",  # "Of course man, I really appreciate this. But it's okay"
            "goemotions-all-042309",  # "Yeah me too, because I would've been a lot more pissed if..."
        ],
        "reasoning": "Quiet positives — light approval, 'cool', 'liked it'."
    },
    {
        "name": "Polite thanks / routine gratitude",
        "description": "Brief, conventional 'thanks' — courteous but low affect.",
        "text_ids": [
            "goemotions-all-045270",  # "ah ok, thanks"
            "goemotions-all-035265",  # "I haven't seen any of them, thanks!"
            "goemotions-all-025078",  # "Thanks mate."
            "goemotions-all-003708",  # "Thanks [NAME], very cool."
            "goemotions-all-007482",  # "Thanks Daily Beast!"
            "goemotions-all-032458",  # "Yes! Bewildering. Thanks for posting. Really."
            "goemotions-all-048842",  # "You're welcome"
        ],
        "reasoning": "Short polite 'thanks' / 'you're welcome' exchanges."
    },
    {
        "name": "Mild interest & curiosity",
        "description": "Asking a follow-up question with no strong affect — seeking info, opening discussion.",
        "text_ids": [
            "goemotions-all-024516",  # "Nice, i have the crown help me and the 309. Can you compare it to those?"
            "goemotions-all-026793",  # "Is it fair to say she's the most versatile makeup artist..."
            "goemotions-all-006362",  # "Damn he told u that huh. What's he think about the [NAME] being back"
            "goemotions-all-000373",  # "whats the [NAME] story ? sorry, not up to date"
            "goemotions-all-009411",  # "Is it really pasta :)"
            "goemotions-all-034158",  # "Oh, I'm dying to know if you noticed any reaction from him!"
        ],
        "reasoning": "Curious follow-ups; some with slight positive lean ('Nice', smileys), mostly informational."
    },

    # ============ MIXED / AMBIVALENT ============
    {
        "name": "Bittersweet / wistful reflection",
        "description": "Mixed positive-negative — fond memory tinged with regret, pity, or longing.",
        "text_ids": [
            "goemotions-all-036288",  # "very wise, your flair reminds me of 'There can be no true despair without hope.'"
            "goemotions-all-036377",  # "Pity because I enjoyed Max Normal."
            "goemotions-all-010742",  # "Doesn't make me tired in the slightest. It makes my joints scream though."
            "goemotions-all-010577",  # "I grew up on the other side of Ama but live in Tulia now. I will have some El Burrito for you"
            "goemotions-all-035353",  # "Don't feel so bad, OP. It's also entirely possible your father was mistaken."
        ],
        "reasoning": "Hold positive and negative simultaneously; 'pity because I enjoyed' is classic mixed."
    },
    {
        "name": "Sympathetic concern / compassion",
        "description": "Expressing care or worry for someone else's misfortune — sorry, caring, offering help.",
        "text_ids": [
            "goemotions-all-030867",  # "Eesh, really sorry for your loss."
            "goemotions-all-050878",  # "Wow I'm sorry. Can you take pictures of your bruises?"
            "goemotions-all-018362",  # "It's interesting that you'd prefer to punish yourself..."
            "goemotions-all-009550",  # "You should have a friend come with you instead to avoid more hurt feelings"
            "goemotions-all-052313",  # "I thought my [RELIGION] boarding school was weird but now it seems pretty tame..."
        ],
        "reasoning": "Reaching out with concern — empathy mixed with sadness at another's experience."
    },
    {
        "name": "Sarcastic / mock-positive criticism",
        "description": "Surface-positive phrasing carrying biting criticism — irony, snark, fake politeness.",
        "text_ids": [
            "goemotions-all-014517",  # "Of course, I could never catch me one of those wonderful incels."
            "goemotions-all-047444",  # "Bam, it's suddenly fixed! Thanks that was great. This is all happening in Bizarro World."
            "goemotions-all-008814",  # "Oh thanks, ~~[NAME]~~ Mitt. We all know you won't provide meaningful opposition"
            "goemotions-all-049242",  # "This sub doesn't like the pro sports teams until they win..."
            "goemotions-all-010564",  # "An online poll that the Berners didn't manage to swarm? Color me shocked."
            "goemotions-all-038171",  # "But they're this close to breaking new evidence of Russian interference!"
            "goemotions-all-021963",  # "Way to cherry pick. Remember that time that..."
            "goemotions-all-038212",  # "Just eat it after. Lol. Im kidding, but it would honestly be the best..."
            "goemotions-all-050877",  # "I'm sure that's a really important issue to you that consumes your daily life..."
            "goemotions-all-053401",  # "Upvoted for visibility. Disapprove of the poll results."
            "goemotions-all-007156",  # "lol eat your own farts Trumpkin"
            "goemotions-all-008814",  # "Oh thanks, ~~[NAME]~~ Mitt. We all know you won't provide meaningful opposition"
            "goemotions-all-042467",  # "At least we will still have broccoli."
            "goemotions-all-046951",  # "Hope [NAME] is watching this but he's probably too busy snorting some lines"
            "goemotions-all-009750",  # "I'm not sure if you're being sarcastic or what but I'm not"
            "goemotions-all-003083",  # "The joke is that people still do this, so it is humorous..."
            "goemotions-all-031576",  # "You sound angry."
            "goemotions-all-003953",  # "The very best they could do was the uranium one unscandal..."
            "goemotions-all-043279",  # "Like protesting for higher minimum wages... we protest for the opposite in Canada."
            "goemotions-all-039367",  # "Because everything is like a video game to them. And in real life the PvP is unbalanced!"
        ],
        "reasoning": "Irony / sarcasm — positive surface words wrapping clear disapproval, mock-naivete, knowing eye-roll."
    },
    {
        "name": "Surprise (unspecified valence)",
        "description": "Surprise / astonishment without clear positive or negative direction — being caught off guard.",
        "text_ids": [
            "goemotions-all-011581",  # "Came here to say that. I'm genuinely astounded that it's as much as 12%"
            "goemotions-all-042468",  # "Predicted almost exactly what was going to happen. Still got me."
            "goemotions-all-019930",  # "There would be a lot of sexual tension in faze..."
            "goemotions-all-052359",  # "I'm surprised he survived the primary."
            "goemotions-all-006688",  # "I had no idea! The art style just seemed similar..."
            "goemotions-all-019028",  # "I'm surprised alcohol isn't entirely sold by the state..."
            "goemotions-all-049727",  # "I've woken up next to some shocking things but just imagine finding..."
            "goemotions-all-018400",  # "Really??"
            "goemotions-all-023971",  # "So you're saying my man was murdered..."
            "goemotions-all-045944",  # "[NAME] H. MYSELF?! He's actually signing that?!"
            "goemotions-all-012786",  # "Holy fuck, that save was on what was essentially a wide open net"
            "goemotions-all-007148",  # "But he got to play with future Hall of Fame QB [NAME]!!!"
        ],
        "reasoning": "Surprise without clear valence — 'astounded', 'really??', 'holy fuck' that save, exclamation of disbelief."
    },

    # ============ NEGATIVE / MILD ============
    {
        "name": "Mild disappointment / let-down",
        "description": "Disappointment of moderate weight — wishing things had gone differently, mild regret.",
        "text_ids": [
            "goemotions-all-001066",  # "Never worked on my babies. So disappointing"
            "goemotions-all-050518",  # "Wish they would of given air vehicles some love. A lot of ace pilots hated the game."
            "goemotions-all-035570",  # "Sloppy ball handling and feels like there's a lid on the bucket. Just not our day"
            "goemotions-all-034417",  # "I know, this weekend was absolute worse case scenario"
            "goemotions-all-035350",  # "That was painful to read."
            "goemotions-all-039843",  # "I think my family would leave TSCC if that happened they are so sexiest"
            "goemotions-all-005522",  # "I'm guessing this is in response to... missed my chance.. contextttt"
            "goemotions-all-011775",  # "as is tradition, guy who played for us scores against us..."
        ],
        "reasoning": "Disappointment register — 'so disappointing', 'wish they had', 'just not our day'."
    },
    {
        "name": "Mild irritation / exasperation",
        "description": "Low-grade annoyance — eye-rolling, mild venting about minor inconvenience.",
        "text_ids": [
            "goemotions-all-041161",  # "Drives me crazy."
            "goemotions-all-008960",  # "[NAME] I hate Twitter"
            "goemotions-all-025447",  # "Tribe member? They aren't native so that's annoying"
            "goemotions-all-025657",  # "Nah, I think it's being as dull as a butter knife"
            "goemotions-all-005164",  # "now that theres mercenaries and its sooo annoying to deal with them"
            "goemotions-all-009845",  # "Ugh. [NAME] is so naughty."
            "goemotions-all-013268",  # "Y'all are fucking annoying"
            "goemotions-all-048082",  # "What a waste of packaging."
        ],
        "reasoning": "'Drives me crazy', 'so annoying', 'ugh'. Lower-arousal than full anger."
    },
    {
        "name": "Mild embarrassment / awkwardness",
        "description": "Cringe, mild self-consciousness, awkward laughter at a faux pas.",
        "text_ids": [
            "goemotions-all-044803",  # "thanks bot! You too man!......Soooooo, what does YOUR hair look like?...."
            "goemotions-all-003606",  # "Sometimes when people ask... I sing this to them and only get confused looks :("
            "goemotions-all-044595",  # "You may have been watching a tad too hard bro lol No judgment here"
            "goemotions-all-046788",  # "It's not my cousin's fault. It was actually me who set up the whole date."
            "goemotions-all-032348",  # "a girl told me I looked like [NAME] from The Little Mermaid. *Ouch*"
            "goemotions-all-018254",  # "Man I've been watching too much Goblin Slayer lately and forgot what sub I was on"
        ],
        "reasoning": "Awkwardness, self-deprecating cringe ('*Ouch*', 'forgot what sub I was on')."
    },

    # ============ NEGATIVE / MODERATE ============
    {
        "name": "Criticism / disapproval",
        "description": "Reasoned negative judgement — calling something wrong, bad, or poorly done. Less heated than anger.",
        "text_ids": [
            "goemotions-all-045768",  # "Do yourself a favor and protect your credibility by not arguing in favor of a terrible hire."
            "goemotions-all-050661",  # "This is an uninformed and incorrect statement."
            "goemotions-all-007399",  # "Warlord actually has NO moveset at all and is literally the most boring videogame char"
            "goemotions-all-038931",  # "How can someone reply without contributing anything meaningful..."
            "goemotions-all-016429",  # "Because our capitalist system is creating ignorant voters by design"
            "goemotions-all-049279",  # "I just didn't think there was something this blatantly immoral in the new testament"
            "goemotions-all-034828",  # "That's not really a solid method."
            "goemotions-all-015161",  # "Incorrect. I'll go with what [NAME] tells me thanks."
            "goemotions-all-007166",  # "Heroin definetley isn't safer than Ritalin..."
            "goemotions-all-004820",  # "All voting has all sorts of problems. Struggling to find a point to your post."
            "goemotions-all-034948",  # "She looks pretty big to me. I doubt she's within a healthy weight range."
            "goemotions-all-031958",  # "Stupid play."
            "goemotions-all-053100",  # "Maybe you're just extremely boring and bland..."
            "goemotions-all-053942",  # "As we shouldn't. We are trying to win a SB here not showcase another star player."
            "goemotions-all-002229",  # "'just be confident' is really hard when you're unattractive"
            "goemotions-all-028150",  # "That's a pretty bad example ngl lol"
            "goemotions-all-045889",  # "would laser lipo or something similar get her to the body shape she wants?"
            "goemotions-all-047973",  # "You cant love this girl if she already has a GF... You'll just ruin a relationship"
            "goemotions-all-049708",  # ">Pay workers more and charge customers less. I mean, they're a publicly traded company"
            "goemotions-all-050812",  # "Unless you're the type of person who rolls through relationships every 3-6 months"
            "goemotions-all-033641",  # ">First of All, Nationalism ≠ Protectionsim. Excuse me, yes, it is."
            "goemotions-all-001714",  # "the Oilers were not picking [NAME] with that draft pick even if we didn't trade it away"
            "goemotions-all-043189",  # "How are his past career stats relevant? Dude hit below the Mendoza line"
            "goemotions-all-023763",  # "This sub: YOU SHOULDN'T SWERVE... WHY DIDN'T SHE SWERE TO AVOID THE COLLISION" (calling out hypocrisy)
        ],
        "reasoning": "Substantive critique without heated profanity — judgmental, calling out, debating."
    },
    {
        "name": "Sadness / regret",
        "description": "Sober sadness — grief, low mood, regret, sense of loss without raw despair.",
        "text_ids": [
            "goemotions-all-020025",  # "Life is just an endless series of disappointments and no one gets out of here alive."
            "goemotions-all-039510",  # "It developed because of my childhood and robbed me of a good bit of good stuff"
            "goemotions-all-038685",  # "Cutting myself after some severe postpartum depression"
            "goemotions-all-003678",  # "I think it's even sadder to go through someone's Reddit account looking for stuff like this"
            "goemotions-all-030181",  # "barely into college... to be this oblivious is pretty sad"
            "goemotions-all-015034",  # "somebody's sad lol lol"
            "goemotions-all-039927",  # "The dying empire."
            "goemotions-all-010749",  # "She was so proud of herself for all the organization and I just felt bad for the kids."
            "goemotions-all-034071",  # "Strangely enough, the student debt still counts. Yet another sign civilization is doomed."
            "goemotions-all-023024",  # "Cos Disneyland is too damn expensive for them these days."
            "goemotions-all-016695",  # "I wish [NAME] son was older so it be possible he did this."
        ],
        "reasoning": "Down-mood, 'sad', 'feel bad', existential weariness, mention of depression/self-harm, lamenting cost / wishing things were otherwise."
    },
    {
        "name": "Frustration / venting",
        "description": "Stronger annoyance than irritation but still controlled — frustration with a situation, complaints.",
        "text_ids": [
            "goemotions-all-028406",  # "Can't believe this game actually released in the state it's in complete joke, should be criminal"
            "goemotions-all-008256",  # "Well good luck if you yourself help with that, i'm out of here, i can't be bothered fighting with USSR stans"
            "goemotions-all-019726",  # "I'm sorry you're too fragile about your own to have reasonable discussion"
            "goemotions-all-007220",  # "This fucken essay"
            "goemotions-all-040947",  # "Meanwhile I came in to a clinic the other week with a swollen toe and left with *codeine*"
            "goemotions-all-047064",  # "the other principal in the original story, has gotten out of this unscathed is annoying"
            "goemotions-all-032285",  # "Why are you so proud of being ignorant?"
            "goemotions-all-032819",  # "[NAME] this is shite. Can we get back to Brexit?"
            "goemotions-all-039520",  # "I can't look at [NAME] with this hair!"
            "goemotions-all-017296",  # "Hangry is real and it comes with a horrible vengeance"
        ],
        "reasoning": "Venting frustration — 'can't believe', 'i'm out', 'this is shite', visceral complaints."
    },
    {
        "name": "Contempt / dismissal",
        "description": "Looking down on someone, telling them off, dismissive insults.",
        "text_ids": [
            "goemotions-all-029615",  # "No one cares"
            "goemotions-all-011950",  # "Stop crying."
            "goemotions-all-033303",  # "Sydney is not for you"
            "goemotions-all-002954",  # "Tell her to stop being so fat, then. Edit: /s" - sarcastic but contempt
            "goemotions-all-028488",  # "Enjoy being a scrub."
            "goemotions-all-039770",  # "Pls shut up"
            "goemotions-all-026428",  # "they now wear these ridiculous waist-high tights to hide..."
            "goemotions-all-031210",  # "The kind of women they'll never get."
            "goemotions-all-050571",  # "Dump her. Shes not committed to you"
            "goemotions-all-051939",  # "You have your opinion and I have mine why does that make only one of us a victim?"
            "goemotions-all-000843",  # "Did you read the post?"
            "goemotions-all-036695",  # "Get your degree first."
            "goemotions-all-027360",  # "No thats [NAME]...."
            "goemotions-all-031210",  # already in - skip
        ],
        "reasoning": "Dismissive put-downs — telling someone to shut up, calling them out, looking down, condescending lectures."
    },
    {
        "name": "Suspicion / distrust",
        "description": "Doubting motives, calling out perceived fakery or hidden agendas, conspiratorial framing.",
        "text_ids": [
            "goemotions-all-000260",  # "We were all duped? I've seen through her from day 1, even posted about how fake she is"
            "goemotions-all-017377",  # "He'll just keep asking questions in bad faith to continue getting your attention"
            "goemotions-all-044135",  # "It starts with the malicious ones... [NAME] is truly a stain on american politics"
            "goemotions-all-019980",  # "may I direct your attention to the aforementioned 'tin foil conspiracy'"
            "goemotions-all-032916",  # "Every time it happens people get accused of view botting."
            "goemotions-all-013890",  # "What is his endgame? People are talking about him but this just makes him look bad."
            "goemotions-all-000313",  # "Source each of these and trend lines... listing a few agencies and telling people to Google 8t isn't a source."
            "goemotions-all-030901",  # "They pushed this story as well."
            "goemotions-all-040832",  # "No way that was word for word. While I admit that wasn't original, I typed that from memory."
        ],
        "reasoning": "Calling out bad faith, fakery, conspiracy framing, accusing of plagiarism / pushing narratives."
    },

    # ============ NEGATIVE / STRONG ============
    {
        "name": "Intense anger / hostility",
        "description": "Heated anger — profanity, hostile insults, rage.",
        "text_ids": [
            "goemotions-all-034862",  # "Pretty fucking brutal when your starting front court is 0-10..."
            "goemotions-all-053580",  # "Lol i got bottled by one of them in a brawl with a bunch of the nazis"
            "goemotions-all-049057",  # "Fascism doesn't even exist in America. [NAME], these kids are so sheltered."
            "goemotions-all-049762",  # off-topic
            "goemotions-all-021951",  # "your mom is safe when she bangs me..."
        ],
        "reasoning": "Direct hostility, 'fucking brutal', tribal insults / inflammatory ribbing."
    },
    {
        "name": "Disgust / revulsion",
        "description": "Visceral rejection — moral disgust at acts or comments, expressions of being grossed out.",
        "text_ids": [
            "goemotions-all-016641",  # "its implying he wants to fuck the dog. The original the dog is just scared of him."
            "goemotions-all-004056",  # "Seeing as how someone was raping her..."
            "goemotions-all-016453",  # "Dude, whenever I'm bored I either masturbate play games or browse reddit..."
            "goemotions-all-021972",  # "someone would have to do some serious mental gymnastics to convince themselves... taking advantage of an incapacitated person"
            "goemotions-all-037779",  # "You didn't want to have sex with anyone before you turned 18?"
            "goemotions-all-025225",  # "But officer, I have a doctor's note saying I *have* to expose myself to them!" -- dark joke but disgust adjacent
            "goemotions-all-017310",  # "two female chiefs being accused of molesting two male trainees"
            "goemotions-all-027475",  # "Grab her testicles"
            "goemotions-all-028331",  # "If you wanna know how rappers rape ask [NAME]"
            "goemotions-all-034617",  # "'You look a little older without your clothes on'"
            "goemotions-all-036614",  # "'but did it feel good tho?'"
        ],
        "reasoning": "Morally repulsive content; lewd/violent imagery; commenters relating with disgust."
    },
    {
        "name": "Intense fear / anxiety",
        "description": "Acute worry, dread, or fear — strong negative anticipation.",
        "text_ids": [
            "goemotions-all-032997",  # "This gave me ultimate anxiety"
            "goemotions-all-002702",  # "So then what do I do? Should I tell priests? Will they kick me out?"
            "goemotions-all-002064",  # "Time to go to heaven, son."
            "goemotions-all-013536",  # "You're now banned from /r/starwars and some YouTube channels"
            "goemotions-all-003117",  # "Don't say his name!!"
            "goemotions-all-025717",  # "Fucking hell..."
            "goemotions-all-045495",  # "you should try to stay cool and dont exert yourself" - cautioning
            "goemotions-all-048641",  # "Yeahhhh I'm going to go to sleep now..."
        ],
        "reasoning": "Anxiety, dread expressions, 'ultimate anxiety', alarmed cautioning, nervous retreat."
    },

    # ============ OTHER AFFECT TYPES ============
    {
        "name": "Hopeful anticipation / aspiration",
        "description": "Positive forward-looking — hope, optimism about a coming event, encouraging perseverance.",
        "text_ids": [
            "goemotions-all-025895",  # "I'm resigning my good paying FT soul sucking job to try something new. Wish me luck!"
            "goemotions-all-045862",  # "Injuries will not knock us out of the running."
            "goemotions-all-048568",  # "I'm an inspiring man"
            "goemotions-all-002687",  # "Now that he has been there, he hopefully knows better than to be wasteful"
            "goemotions-all-010212",  # "Everyone's gotta face the music some day brother"
            "goemotions-all-012120",  # "I would pay good coin to read that love story."
            "goemotions-all-022505",  # "We should sign him to a 4.5AAV contract now that he has already proved himself as an elite starter"
            "goemotions-all-001321",  # "Hey! Any chance you still have those templates? I'd love to replace my screens..."
            "goemotions-all-031434",  # "Better still get the free game!"
            "goemotions-all-048994",  # "I'm really high on [NAME] from Wisconsin"
        ],
        "reasoning": "Forward-looking optimism — wishing luck, hopeful about future, enthusiastic recommendations."
    },
    {
        "name": "Confusion / bewilderment",
        "description": "Not understanding, asking what just happened, genuine puzzlement.",
        "text_ids": [
            "goemotions-all-005105",  # "Character and emotions are not the reigning champion one year earlier..."
            "goemotions-all-012221",  # "That's ruis intentions"
            "goemotions-all-003323",  # "Is this supposed to be an ice cream sundae or mashed potatoes with gravy?"
            "goemotions-all-029518",  # "There's no L in cyrillic. They were so close."
            "goemotions-all-024983",  # "Why would you cross while an event is going..."
            "goemotions-all-022376",  # "It was so ballsy. I still shake my head, thinking about how....tactless it was."
            "goemotions-all-009054",  # "Yeah. I heard that in my head aswell doing the last argument..."
            "goemotions-all-007811",  # "I will try to pay better attention. Its definatly not just ambient noise."
            "goemotions-all-016689",  # "I thought for a moment the one mongoose was dead..."
            "goemotions-all-029210",  # "Well, seeing as how communism is by definition stateless..."
            "goemotions-all-010661",  # "So [NAME] guided the process of canonization but then left?"
            "goemotions-all-025133",  # "It's better for the future to suck and draft top 5 than be mediocre and draft 12th?"
            "goemotions-all-019977",  # "Putting the opposite outcome of what [NAME] says is a close second imo"
            "goemotions-all-008128",  # "we don't really know"
            "goemotions-all-010889",  # "ye but that is fucking [NAME]... there is a diffrence between [NAME] and [NAME]"
            "goemotions-all-008450",  # "Meh, same branch of religion. Just opposing views on the sequel to the story."
            "goemotions-all-010893",  # "I do, and I've experienced a similar situation falling asleep. Like my mind is fixated on an object"
            "goemotions-all-030099",  # "I guess I should specify. I would like a 'right' and 'left' political viewpoint."
        ],
        "reasoning": "Puzzled / asking 'why' / can't parse / shaking head / clarifying because misunderstood."
    },
]

# Collect assigned IDs and detect conflicts
assigned = []
cluster_by_id = {}
for c in clusters:
    # dedup within
    c["text_ids"] = list(dict.fromkeys(c["text_ids"]))
    for tid in c["text_ids"]:
        if tid in cluster_by_id:
            print(f"DUPLICATE: {tid} in {cluster_by_id[tid]} and {c['name']}")
        cluster_by_id[tid] = c["name"]
        assigned.append(tid)

assigned_set = set(assigned)
unassigned = sorted(all_ids - assigned_set)

# Verify all assigned IDs exist
for tid in assigned_set:
    if tid not in all_ids:
        print(f"UNKNOWN ID: {tid}")

print(f"Total assigned: {len(assigned_set)}  Clusters: {len(clusters)}  Unassigned: {len(unassigned)}")
print("Cluster sizes:")
for c in clusters:
    print(f"  {len(c['text_ids']):3d}  {c['name']}")

# Write the proposal
ts_compact = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
ts_iso = datetime.datetime.utcnow().isoformat() + "Z"
proposal = {
    "timestamp": ts_iso,
    "sample_size": len(sample),
    "sample_strategy": "shared_ids",
    "style": "valence-intensity",
    "existing_clusters_considered": False,
    "clusters": clusters,
    "unclustered_ids": unassigned,
    "observations": (
        "Organized by valence (positive / mixed / negative) crossed with intensity "
        "(mild / moderate / strong) and then split each cell into the affect "
        "subtypes that actually appear in the sample. Sample distribution skews "
        "moderately negative — criticism, mild irritation, and contempt are large "
        "cells — but positive admiration and amusement are also well-populated. "
        "Sarcastic mock-positive is its own cell because the surface lexicon would "
        "otherwise force it into positive admiration. Surprise and confusion are "
        "broken out separately because their valence in this sample is genuinely "
        "ambiguous. Unclustered comments are either purely factual/informational "
        "with no detectable affect, or so context-dependent (sports lineups, "
        "draft picks, named individuals with no emotion lexicon) that no primary "
        "emotion can be inferred from the text alone."
    ),
}

prop_dir = WORKSPACE / "proposals"
prop_dir.mkdir(parents=True, exist_ok=True)
prop_path = prop_dir / f"prop_{ts_compact}_pbbb.json"
with open(prop_path, "w", encoding="utf-8") as f:
    json.dump(proposal, f, indent=2, ensure_ascii=False)
print(f"\nWrote {prop_path}")
