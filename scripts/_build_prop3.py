"""Build proposer #3 (humanities/lifestyle/games lens) proposal JSON.

Reads sample_proposer3.json from the workspace, builds a cluster mapping,
validates that every text ID is assigned exactly once, and writes the
proposal file.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(
    "C:/Users/emily/Documents/agentic-clustering/results/clustering/stackexchange/seed=0"
)
SAMPLE_PATH = WORKSPACE / "sample_proposer3.json"

# Each cluster: (name, description, reasoning, [ids])
CLUSTERS: list[tuple[str, str, str, list[str]]] = [
    # ============================================================
    # GAMES (split fine-grained)
    # ============================================================
    (
        "Tabletop RPGs (D&D / Pathfinder / WoD rules)",
        "Rules and mechanics questions about tabletop role-playing games such as D&D, Pathfinder, World of Darkness, etc.",
        "Multiple texts ask about RPG mechanics (Wrathful Aspect, returning weapons, WoD version, restarting campaign, resurrection of dead characters).",
        ["1872", "3555", "1000", "1081", "4116"],
    ),
    (
        "Video games: specific titles (Star Trek Online, League of Legends, OpenTTD, Alien Swarm, Steam OS)",
        "Gameplay questions about specific named video games and gaming platforms.",
        "Star Trek Online science officer, League of Legends lanes, OpenTTD transfer, Alien Swarm screenshots, Steam OS motherboard.",
        ["2282", "425", "3828", "2269", "1942", "2464", "2719", "684"],
    ),
    (
        "Retro / vintage computing games",
        "Questions about classic / vintage gaming hardware and old game titles (Apple II, NES, TMS9918).",
        "Three texts: Elite on NES, Apple II timing resistor, TMS9918 chip evolution.",
        ["43", "730", "1275"],
    ),
    (
        "Card / poker games",
        "Questions about card games and poker hand mechanics.",
        "Player declares at the river, bet sizing.",
        ["1260", "4078"],
    ),
    (
        "Riddles, puzzles and word games",
        "Riddle posts and puzzle/word-game style challenges.",
        "Rhyming Riley, Riley Riddle with ciphering, halve and race to unity, hidden meaning, identity riddle, 'you have played with me before'.",
        ["1235", "3825", "81", "2579", "2354", "3009"],
    ),
    # ============================================================
    # LANGUAGE LEARNING — by target language
    # ============================================================
    (
        "Japanese language learning (grammar/usage)",
        "Questions about Japanese grammar, vocabulary, particles and meaning.",
        "Many texts ask about Japanese grammar/words: と meaning, 気をつけて, は/を, くらい/ほど, ~도 ~나 ~ㄹ까 (Korean), 世界/世, きない/こない, ドリフ, onomatopoeic sound.",
        ["3086", "1956", "130", "1029", "4080", "743", "2516", "718"],
    ),
    (
        "Korean language learning",
        "Korean grammar/usage questions.",
        "Korean pattern ~도 ~나 ~ㄹ까.",
        ["56"],
    ),
    (
        "Spanish language learning (grammar/translation)",
        "Spanish grammar, idiom, vocabulary and translation questions.",
        "Multiple Spanish posts: 'siempre quieren', 'de nana con la pelota', 'transporte es por carrera', 'sans' (cross-lang), 'escalar/hacer escala', 'leading him/her on' in Spanish, redundant pronoun.",
        ["1307", "844", "1379", "919", "1209", "3382"],
    ),
    (
        "Portuguese language learning",
        "Portuguese grammar and translation questions.",
        "How to say 'let's' in Portuguese; 'cramming' termo equivalente; artigo correto em palavras em outras linguas.",
        ["2865", "2561", "1027"],
    ),
    (
        "German language learning",
        "German grammar, vocabulary, and translation questions.",
        "Many German-language posts: Heia origin, der/die Chicorée, Ausstellungs vs Erteilungs, Tanzverbot translation, Praktikum types, Perfekt vs Präteritum, sprachliche Konstrukte richtig/falsch, in/im Lotto, 'across a crowded room'.",
        ["1959", "3314", "695", "1011", "1800", "840", "2232", "3010", "1274"],
    ),
    (
        "Italian language learning",
        "Italian grammar, idiom and dialect questions.",
        "'Barbellare' dialect, mi dispiace vs scusa, milza meaning, country road term, 'little Willie' Italian equivalent, -che conjunctions.",
        ["460", "1369", "3517", "3682", "1030", "2119"],
    ),
    (
        "Russian / Ukrainian / Slavic language learning",
        "Grammar/usage and meaning questions in Russian, Ukrainian, and related Slavic languages including Old Church Slavonic.",
        "Russian comma after 'хотя', 'был крутой/крутым', emoji debate, pronoun-noun agreement, NE separately or together, 'пункт' in Old Slavonic, Ukrainian вдуплятися, німами, different 'number' in verb of relative clause using который.",
        ["2112", "2820", "3413", "287", "2847", "4009", "2474", "3760", "2652"],
    ),
    (
        "Esperanto / constructed and minority languages",
        "Questions about Esperanto and other constructed/minor languages.",
        "Esperanto article use 'La signifo de forlaso de la artikolo'.",
        ["2391"],
    ),
    (
        "French language learning",
        "French language grammar, translation and meaning questions.",
        "'s'il te/vous plaît' French and Latin.",
        ["2855"],
    ),
    (
        "Arabic language",
        "Questions about Arabic dialects and Arabic-language linguistics.",
        "Most understandable Arabic dialect.",
        ["1292"],
    ),
    (
        "English grammar, usage, etymology",
        "English-language grammar, usage, term/idiom, etymology, and translation questions.",
        "Many: 'sans' article, term in statement, gossip word, sheet music vs musical sheet, suffix -sal, Xmas24, postpositional clauses, complex vs leading question, 'connected vs interconnected', pronouns for ghosts, makeshift bookend term, 'old German' script name, computer-manipulated gibberish term, 'kiss of death' meaning, two melodies term, pronunciation difference, gossiper word, time spell-out, indefinite relative pronouns, line-end X long syllable, 'beata' translation, adjectives from numerals, hand-translation, 'A new world awaits' translation, 'I have a bad accent'.",
        ["1149", "480", "929", "1984", "180", "254", "1552", "1075", "2073", "2910",
         "2441", "1325", "3667", "1266", "1443", "1050", "131", "2797", "115", "887",
         "1855", "545", "2996"],
    ),
    (
        "Linguistics, corpus linguistics, manuscripts",
        "Theoretical linguistics, ancient manuscript study, and language-corpus questions.",
        "Finite set of meaning blocks, ancient manuscript transcription, linguistic corpus pricing, Hebrew/Greek Bible textual variants.",
        ["2339", "1556", "1035", "1023"],
    ),
    (
        "Language-learning resources & courses",
        "Resources, sites, tools, and methodology for learning languages.",
        "Site with video-based language courses.",
        ["1593"],
    ),
    # ============================================================
    # HUMANITIES — RELIGION & PHILOSOPHY
    # ============================================================
    (
        "Christianity (Bible interpretation & theology)",
        "Questions about Christian theology, biblical passages, and church history.",
        "Folded napkin in empty tomb, casting lot in Acts 1:26, St Peter as Rome bishop, Abraham's sacrifice in Genesis, Shechem to Dosan, Bhagwad Gita killing (Hindu but religion).",
        ["396", "2841", "2773", "2809", "3324"],
    ),
    (
        "Judaism (halacha, Torah, Jewish thought)",
        "Questions about Jewish law (halacha), Torah teaching, ritual practice, and Jewish philosophy.",
        "Da' Mah LeHashiv vs teaching Torah to non-Jews, Bishul Akum intent, 'Jewish Philosophy' by halacha.",
        ["3721", "3362", "2569"],
    ),
    (
        "Islam (fiqh, ritual practice)",
        "Islamic ritual, theological, and fiqh questions.",
        "Wudu head cover, items considered najis, ruling on prayer without attention.",
        ["1827", "2632", "1501"],
    ),
    (
        "Hinduism, Vedas, Indian religions",
        "Questions about Hindu scriptures, ethics, and Indian religious traditions.",
        "Bhagavad Gita justified killing, Rigveda basic facts.",
        ["58", "3411"],
    ),
    (
        "Mythology (Greek, Roman, world)",
        "Questions about classical mythology and deities.",
        "Deities diminished by Greek pantheon, Cadmus' wedding by Olympians, Muses governing arts.",
        ["1754", "795", "3057"],
    ),
    (
        "Philosophy (general / logic / ethics)",
        "Questions about philosophical arguments, logic, metaphysics, ethics, and rating essays.",
        "Step in argument least valid, Kant on spider moms, temporal logics paraconsistent, derive sequent, rate philosophy essays, robot programmed human, absence of negative vs presence of positive feedback.",
        ["2435", "1414", "3410", "2611", "3542", "3330", "1110"],
    ),
    # ============================================================
    # POLITICS / LAW / ECONOMICS / FINANCE
    # ============================================================
    (
        "Politics / government / public policy",
        "Questions about political institutions, elections, and government policy.",
        "Canadian mayor parties, Congress concurrent state office, Fair Labor Standards Act citations.",
        ["2885", "1990", "3425"],
    ),
    (
        "Economics (theory, policy, central banks)",
        "Theoretical and applied economics, including monetary policy and macro.",
        "Currency pegging & FX reserves, IoT and Bosch cooling costs, universal basic income & Marxist exploitation.",
        ["2220", "1711", "3913"],
    ),
    (
        "Personal finance / taxes / retirement",
        "Personal finance, taxes, pensions, and retirement-related questions.",
        "IRS Letter 96C, pay self employed, taxes quit-claimed home, pension fund switch, employment freelancing rates, recruitment consultant payment, paid trial work contract.",
        ["1132", "1008", "2047", "3154", "709", "189", "1980"],
    ),
    (
        "Quantitative finance / trading / options",
        "Questions about derivatives, options, hedging, futures, technical finance.",
        "Bond futures hedging duration, risk-neutral probability binomial, Yahoo Finance NYMEX ticker, stop vs ask order, put/call vs futures, hedge BEKK in R, replicating Yahoo Beta R, GARCH(1,1) prediction in R.",
        ["3271", "1891", "1326", "769", "1091", "2429", "1247", "2167"],
    ),
    (
        "Law / patents / legal procedure",
        "Patent, IP, and legal procedure questions.",
        "Manufacture patented but discontinued, prior art creation without filing, Capital One targeted promotions prior art request.",
        ["3213", "3621", "985"],
    ),
    # ============================================================
    # TRAVEL / IMMIGRATION / VISA
    # ============================================================
    (
        "Travel destinations, advice, money abroad",
        "Travel destination tips and practical advice (where to go, ATMs, currency, shopping abroad).",
        "Tokyo things to do, Mongolia ATMs Cirrus, biggest European hospitals, Kirschwasser in Frankfurt.",
        ["3670", "3253", "2210", "3265"],
    ),
    (
        "Visas, immigration, citizenship",
        "Questions on visas, residence permits, citizenship, and immigration documents.",
        "Australian tourist visa for Philippine, French nationality grandfather army, China license expired original, French resident permit travel Europe, ACS skill assessment, B2 long stay unmarried partner, citizenship by descent, getting back renounced citizenship.",
        ["1117", "4140", "1603", "3605", "1273", "3820", "1056", "3405"],
    ),
    # ============================================================
    # COOKING / FOOD / BREWING / GARDENING
    # ============================================================
    (
        "Cooking, food science, ingredients",
        "Cooking techniques, food science, recipes, food safety questions.",
        "Glycemic index research, apple seeds arsenic, dairy mucus, body temp & sleep (overlaps health), inflammation grounding (health).",
        ["2817", "1285", "2950"],
    ),
    (
        "Homebrewing, beer, fermentation",
        "Homebrewing/beer-making questions including yeasts, recipes, BJCP categories, and bottling.",
        "Add more yeast & sugar, yeast strains for warm/hot fermentation, BJCP Black Witbier, grain in 16L kettle, barleywine aging, juniper for Sahti, recap commercial bottles.",
        ["2536", "477", "1514", "1362", "682", "4007", "3075"],
    ),
    (
        "Gardening / home & yard",
        "Outdoor plants, hedges, lawn / yard care.",
        "Boxwood hedge paver walkway proximity.",
        ["3830"],
    ),
    # ============================================================
    # HOME / DIY / WOODWORKING / HVAC
    # ============================================================
    (
        "Woodworking and joinery",
        "Joinery, dovetails, tenons, machining wood.",
        "Tusked tenon dowel, sliding dovetail thickness, temporarily attach wood for machining.",
        ["1296", "481", "2583"],
    ),
    (
        "Home improvement / HVAC / construction",
        "Home/building construction, plumbing, heating, waterproofing.",
        "Waterproofing first floors, rocket mass heater chimney, wall heater HDMI interference, tri-clamp sanitary, hydrocyclone (industrial).",
        ["2297", "531", "3302", "419", "2403"],
    ),
    # ============================================================
    # PARENTING / RELATIONSHIPS / INTERPERSONAL
    # ============================================================
    (
        "Parenting, child development, schooling",
        "Parenting questions about discipline, homework, school readiness, infant care.",
        "Parent-child education tie, indecisive 2.5-yr-old, encourage homework, child fail preschool/kindergarten, pushing child too hard, English present for non-English child.",
        ["804", "1747", "3031", "1341", "3905", "1582"],
    ),
    (
        "Interpersonal / family / emotional advice",
        "Personal advice on relationships, family conflict, emotional and social situations.",
        "Close-minded elderly relatives, sharing vs taken advantage of, friend tracking lifestyle, follow up after loss, anxiety alternative phrasing, breastfed once marriage question, conflict with colleague, dating team members procedure, miser/spend more.",
        ["382", "3943", "2787", "787", "376", "402", "2856", "4125", "3148"],
    ),
    # ============================================================
    # PETS / ANIMALS / OUTDOORS
    # ============================================================
    (
        "Pets (cats, dogs, etc.)",
        "Pet care behavior and physiology questions.",
        "Cat bites face affectionate, chihuahua pees excited, water lose scent (dogs).",
        ["2625", "555", "3983"],
    ),
    (
        "Outdoors, hiking, camping, climbing",
        "Backpacking, hiking, climbing, outdoor gear, bears.",
        "Eastern vs Western US backpacking, climbing shoe resoling, at-home routine climbing, camp disturbed area, cross bear, krazy glue outdoor gear, figure-of-8 SRT, improvised ice scraper.",
        ["1519", "1511", "2226", "865", "2465", "3168", "847", "832"],
    ),
    # ============================================================
    # MUSIC
    # ============================================================
    (
        "Music theory & playing instruments & audio production",
        "Music theory, scales, performance, instruments, sound editing, audio production.",
        "Scale solo octave fretboard, sound editing for battle scenes, Juno-Gi & Drumbrute sync, song composers melodies, sfx timeline spotting.",
        ["3587", "2957", "3746", "3193", "571"],
    ),
    (
        "Identifying / finding music",
        "Help identifying songs or finding music.",
        "Early 2000s pop song hook, Battle of Kiska song full text.",
        ["13", "3976"],
    ),
    # ============================================================
    # MOVIES / TV / FICTION / WORLDBUILDING
    # ============================================================
    (
        "Movies / TV / film identification",
        "Movie/TV questions and identification.",
        "Josey Wales spitting, 80s movie gun freezes people, Netflix descriptions.",
        ["739", "1142", "507"],
    ),
    (
        "Sci-fi & fantasy lore (Star Wars, Star Trek, MCU)",
        "Questions about lore of speculative fiction franchises.",
        "Jabba Hutt's pet, Federation starships count, Superman physiology, Scarlet Witch imprisonment.",
        ["1760", "2125", "2035", "2412"],
    ),
    (
        "Worldbuilding / speculative scenarios",
        "Creating fictional worlds and physics-of-scenarios questions for stories.",
        "Forest cities galaxy civilization, contact with God via micro black hole story, dinosaur-humans coexistence planet, habitable binary star planet, moon eating world, fake a dead body.",
        ["2164", "2623", "3470", "2821", "3677", "2875"],
    ),
    (
        "Writing / screenwriting / literature craft",
        "Craft questions for fiction writers, screenwriters, poets.",
        "Phone scene screenplay, internal monologue characters, citing other media in writing, learning poetry benefits, write Wikipedia edit, hyperbaseurl pdf, two chapters one page, Pinocchio nose paradox.",
        ["1308", "2853", "3220", "3237", "1099", "4044", "2935", "4107"],
    ),
    # ============================================================
    # PHOTOGRAPHY
    # ============================================================
    (
        "Photography (cameras, lenses, settings)",
        "Camera, lens, exposure, film, and post questions.",
        "Nikon D600 mode wheel, Samyang vs Rokinon fisheye, underexposed film, pricing prints, Nikon 5100 photo details, JPG quality detection, low f-number lenses, EOS / sensors.",
        ["1077", "3142", "2401", "2751", "2485", "4067", "2674"],
    ),
    # ============================================================
    # ART / DESIGN / GRAPHICS
    # ============================================================
    (
        "Graphic design / typography / fonts",
        "Visual design, fonts, typography, layout, print color management.",
        "Web safe font Chinese, title page front matter, dvips EPS bounding box, mono fonts, CMYK Adobe PDF, RTF embedded images, free design work, choose right chart.",
        ["554", "2510", "1195", "3793", "1925", "381", "1382", "2495"],
    ),
    # ============================================================
    # PROGRAMMING / SOFTWARE — by language/platform/framework
    # ============================================================
    (
        "PHP / WordPress / scripting-language web programming",
        "PHP, WordPress and PHP-script issues; non-English-language PHP/JS/script programming Q&A.",
        "PHP escape on JS, PHP session jquery, WordPress plugin android, PHP SQL parsing, calling AGPL via PHP, simulate date PHP EE, Form não passa via POST, script returning object, JS doc IDE, while loop bucle, tree element fill, type conversion error, Russian textarea printing, Russian 'Игра' coding task.",
        ["2389", "243", "1226", "2187", "792", "244", "2454", "3639", "3796", "824", "2351", "3798", "2424", "1106"],
    ),
    (
        "JavaScript / jQuery / CSS / frontend dev",
        "Frontend web programming with JS, jQuery, CSS, HTML.",
        "Best practices external JS, JS/jQuery UI/UX, JS/jquery tic tac toe, populate Href, hyperlink target window, browser-cached CSS/JS, responsive rotator, 960 Grid System, select element multiple classes.",
        ["2511", "302", "3212", "1503", "830", "1922", "2408", "1156", "2214"],
    ),
    (
        "Java / Scala / JVM languages",
        "Java/Scala-language programming questions.",
        "AudioInputStream getResource error, Java cross-platform language choice, eclipse oxygen scala IDE HelloWorld, merge PDFs JAVA, AlarmManager Notificação, passar valor para função em java.",
        ["1899", "1986", "829", "392", "432", "2436"],
    ),
    (
        "C# / .NET / ASP.NET / VBA",
        "Programming in C#/.NET, ASP.NET Web API, Excel VBA.",
        "Embeddable web server WinForms ASP.NET, dynamic types in C#, MongoDB C# (overlap SQL), ASP.NET 4.0 sharepoint (overlap), Excel VBA clear selection.",
        ["3880", "532", "2596"],
    ),
    (
        "Python / R / data-science / ML",
        "Python or R for stats / ML / data science / ImageHistogram / Mathematica.",
        "Random forest R, statistical tests in R, ImageHistogram mean stddev, Hessian-free, Fisher Neyman-Pearson, predicting time interval statistics, LiDAR ICP.",
        ["1205", "3225", "168", "3977", "2981", "918", "3328"],
    ),
    (
        "C / C++ / low-level systems programming",
        "C/C++, memory bugs, OpenCV, eigenvalues compute.",
        "Buffer overflow function pointer, hardware memory leaks, C++ interview questions, opencv padding issue, ARPACK eigenvalues compute time.",
        ["3598", "467", "1210", "2051", "426"],
    ),
    (
        "Other programming languages (Go, Scheme, Lisp)",
        "Languages like Go, Scheme, Common Lisp, etc.",
        "Account manager Go transactions, Scheme vs Common Lisp.",
        ["513", "3538"],
    ),
    (
        "Vim / editor / shell scripting",
        "Editor (Vim), shell, batch, cron tooling.",
        "Remove semicolon Vim, vim markdown indentation, moving to first letter, batch file link params, git push crontab.",
        ["3554", "1436", "1631", "4089", "1041"],
    ),
    (
        "SQL / databases / data engineering",
        "Database queries, MongoDB, Oracle, MySQL data ops, Apache logs to CouchDB.",
        "Database.query string, oracle CSV export, MySQL random selection, two daemons same DB, MongoDB+C# (2158), Apache HTTPd to CouchDB (3637).",
        ["431", "2747", "3756", "1401", "2158", "3637"],
    ),
    (
        "Web development general (architecture, APIs, SEO, search tricks)",
        "Cross-language web dev concerns: SEO, APIs, web architecture, embed, search-engine tricks.",
        "Embed link track + SEO, API user requests, Postman header, JSON post support, stop Google crawl, SEO duplicate product, retrace SEO redirect, sub-domain to sub-domain, Google search day given date.",
        ["2848", "2415", "1689", "3346", "3693", "3687", "1478", "875", "4063"],
    ),
    (
        "SharePoint / enterprise CMS (Sitecore, Magento, EE, Joomla)",
        "Enterprise CMS platforms — SharePoint, Sitecore, Magento, ExpressionEngine, Joomla — admin/config/customization.",
        "Many SharePoint/Sitecore/Magento/EE/Joomla specific issues.",
        ["458", "82", "1361", "1310", "552", "179", "2754", "2988", "3406", "3344",
         "2420", "2384", "546", "1317", "2858", "2673", "192", "1278", "2657",
         "1557", "3744", "3556", "1063", "3995", "1530", "3770", "2648", "3582", "2313", "4124"],
    ),
    (
        "Stack Exchange / StackApps / meta tools",
        "StackExchange / StackApps platform internals, meta tools, API.",
        "StackApps plugin, paging 10k search, get user vote counts, GM script revert, edit URL for tag wiki, clipboardy chrome, status of OKFN Open Product, sign-up email feedback, automated word filter.",
        ["2523", "1842", "3871", "1550", "3945", "1541", "3602", "3312", "823"],
    ),
    (
        "Web/network/system security",
        "Security questions on websites, networks, OS, browsers, sessions.",
        "Sensitive doc storage, malicious bots spam, IP whitelist IP camera, secure hyperlinks, decoy relay user, port opening gateway, Symfony2 sessions, email validation, torrent SHA, RADIUS for VPN, prevent program internet, port subnet, 2FA removal Stellar, e-mail validation.",
        ["257", "2598", "1871", "565", "2045", "283", "1419", "154", "2655", "1600", "278", "616", "4057"],
    ),
    (
        "Networking, DNS, internet connectivity",
        "Networking issues — DNS, IP, modem, routing, ports.",
        "Modem disconnect Ubuntu, default gateway, DNS geo-redundant, Cisco spanning tree, RaspberryPi crashes, Firefox/Chrome reach site.",
        ["878", "2393", "1564", "1960", "3562", "848", "564"],
    ),
    (
        "Operating systems / installation / drivers / mobile-OS",
        "OS-specific issues — Windows, Linux/Ubuntu, Mac, VMs, drivers, peripherals, mobile OS quirks.",
        "Ubuntu modem (also nw), IRIX Indy, Win7 VMware sound, transfer media windows7-windows phone, VirtualBox sound, X11 fullscreen Mac, HPUX setacl, Win7 system tray, Linux time dilation, Win phone notif, Xbian harddisk, Chrome Android versions, SD card died, Airport extreme HDD, phone beep on missed call.",
        ["2302", "4068", "514", "1355", "1515", "2030", "4000", "3457", "2038", "1877", "2736", "3920", "2537", "1499", "746", "3329"],
    ),
    (
        "Software engineering / SDLC / project management",
        "Software engineering practices: design patterns, testing, agile, PM, data-integrity reasoning.",
        "Cross-functional teams study, design patterns vs algorithm, ISTQB complete testing, continuous integration PO sprint, BPM agile SCRUM, MS Project compare, WBS charge number, exception sanity check, report performance, PMP application, selenium training, code review tracking, data stream confirmation/corruption.",
        ["12", "1192", "701", "226", "3118", "660", "2025", "928", "2234", "2945", "3480", "4008", "3715"],
    ),
    (
        "Source control / Git / Wikis",
        "Version control, Git/GitHub, wikis, editing collaboration.",
        "Git for multiple themes, add modified data to wiki pages, svn update SSL handshake.",
        ["753", "33", "3545"],
    ),
    # ============================================================
    # SCIENCE - PHYSICS / CHEMISTRY / MATH / ASTRONOMY
    # ============================================================
    (
        "Mathematics (pure & applied)",
        "Pure math, analysis, differential equations, abstract algebra.",
        "Continued fractions equations, constants count, linear diff eq teaching, partial deriv integrals commute, modular eq 324x mod 121, anti-groups/anti-manifolds, ln decreasing function, copies to enlarge array, absolute values not centered, commutator operator.",
        ["2368", "62", "1115", "1229", "1718", "2434", "2941", "1353", "1134", "1300"],
    ),
    (
        "Physics (classical / quantum / relativity)",
        "Physics — classical mechanics, EM, quantum mechanics, GR.",
        "Runge-Lenz vector, force free space dielectric, gravitational sphere cavity, condition on classical bit (quantum computing), mass matrix Lagrange, ideal vs actual Rankine cycle, yield strength Fe crystal, methane Titan heat source (also astro), stepper motor weight (engineering), quadcopter lift (engineering), mechanical integrator.",
        ["3662", "3619", "584", "1617", "2136", "762", "3189", "1533", "1625", "1673"],
    ),
    (
        "Astronomy / space / aerospace",
        "Astronomy, space exploration, spacecraft engineering.",
        "Methane Titan heat, payload fairing cost, BA-330 heat rejection, COO milestones (Stellar is crypto, exclude), starships, habitable planet (worldbuilding).",
        ["116", "1679", "2546"],
    ),
    (
        "Skeptics / claims & myths (health, paranormal, etc.)",
        "Claims that need verification — wellness myths, urban legends, conspiracies.",
        "Body temp & sleep quality, grounding/earthing inflammation, cell phones at gas station, flat-earther response, water lose scent dogs (overlap pets).",
        ["4110", "3420", "3834", "3361"],
    ),
    # ============================================================
    # HEALTH / FITNESS / MEDICINE
    # ============================================================
    (
        "Health / medicine (lay / treatments / conditions)",
        "Lay health questions on medications, conditions, treatments.",
        "Silicone for scars, blood donation GM diet, bug in eyes, bedbugs removal, herbal chronic prostatitis, retainers comfort, supplement glutamine, mucus dairy (overlap), grounding (skeptics).",
        ["907", "94", "3147", "3972", "4120", "1340", "3845"],
    ),
    (
        "Fitness / exercise / sports training",
        "Fitness, hypertrophy, HIIT, cardio, exercise training.",
        "HIIT burst rest, hypertrophy 2 vs 3 sets, LBM with cardio.",
        ["2646", "3847", "3788"],
    ),
    (
        "Psychology / mental health / mind",
        "Psychology, behavior, mental health questions (non-clinical).",
        "Run human brain, color-blind suggestion.",
        ["1821", "1605"],
    ),
    # ============================================================
    # MILITARY / HISTORY
    # ============================================================
    (
        "History (military, ancient, modern)",
        "Historical questions — wars, empires, ancient peoples, military history.",
        "Roman General short time, USA-USSR breakdown 1945, Germany division pre-arranged, Bloody Corner, air marshals, Still on patrol origin.",
        ["3136", "3761", "2235", "1", "1609", "3651"],
    ),
    # ============================================================
    # HOBBIES / GAMES BLEND / GEOG / GIS
    # ============================================================
    (
        "Geography / maps / GIS (ArcMap, shapefiles)",
        "GIS, ArcMap, geospatial data and shapefiles.",
        "ArcMap symbology, dynamic layer shapefiles, delete polygon point, Bharatpur landbase, LiDAR ICP (overlap science).",
        ["611", "3041", "984", "851"],
    ),
    (
        "Sports & martial arts (non-video, non-fitness)",
        "Traditional sports — soccer/cricket/etc. — and martial arts.",
        "Goalkeeper handle ball, wicket-keeper stumping, soccer acting, karate style name.",
        ["778", "1007", "3319", "3208"],
    ),
    # ============================================================
    # ENGINEERING / VEHICLES / RADIO / HOBBY ELECTRONICS
    # ============================================================
    (
        "Amateur radio / RF / antennas",
        "Ham radio, Baofeng, RF gear, Military Amateur Radio Service.",
        "Baofeng packet radio, Baofeng BF888s Midland GXT1000, transmit two freqs, MARS.",
        ["3190", "3083", "901", "3675"],
    ),
    (
        "Automotive / motorcycles",
        "Cars, motorcycles, engines, repair.",
        "Hybrid aesthetic wheel covers, motorcycle starter, power steering pump, Ducati desmodromic valves.",
        ["3801", "3906", "3501", "2765"],
    ),
    (
        "Bicycles / personal mobility",
        "Bicycles, gear systems.",
        "Gyroscope/Powerball gearing for bikes.",
        ["1204"],
    ),
    (
        "Embedded / electronics / IoT / Raspberry Pi",
        "Hobby electronics, RaspberryPi, IoT, sensors, motors.",
        "Raspberry Pi wireless 500m, cloud monitor non-event (IoT), DC gearmotor, RC car wheels, robotic teleoperation, hardware memory leaks (overlap), HDMI heater interference (overlap), monitor sometimes thingspeak.",
        ["3745", "4045", "1623", "2582", "950"],
    ),
    # ============================================================
    # MISC HOBBIES
    # ============================================================
    (
        "Cryptocurrency / blockchain (Stellar, Monero, ETH)",
        "Cryptocurrency wallets, networks, smart contracts.",
        "Monero address delete, Cake wallet to MyMonero, ERC20 token wallet, Stellar dividend, Stellar tx_bad_seq, Stellar post-COO sync, payment ID min length, 2FA Stellar (overlap security).",
        ["1264", "2453", "1612", "1931", "2020", "2890", "2154"],
    ),
    (
        "Genealogy / family research",
        "Genealogy and family history research questions.",
        "Death info middle name shown.",
        ["3753"],
    ),
    (
        "Academia / research / grad school career",
        "Academic life, grad school, research practices, internship/job.",
        "Negotiate intern to salaried.",
        ["3455"],
    ),
    (
        "Workplace / career advice (non-academic)",
        "Workplace situations, job offers, resignation, conflict.",
        "Employer won't accept resignation, paid trial work contract (overlap PF), follow up loss (overlap interpers).",
        ["3174"],
    ),
    # ============================================================
    # MISC TECH / PROGRAMMING DETAILS
    # ============================================================
    (
        "Skeptics: data verification & sources",
        "Data lookup and verification — economic stats, hospital data, thermostat data.",
        "Cross country thermostat data, Amazon price history download, Open Product Data status (overlap meta).",
        ["2455", "3947"],
    ),
    (
        "UX / interface design",
        "User experience and UI design questions.",
        "UX/UI vs front-end, title placement mobile, ASC/DSC intuitive sort, contact form correct reason, highlight menu items, optional product cart, data visualization choosing chart (overlap design).",
        ["176", "2802", "551", "1126", "3505"],
    ),
    (
        "Project / process methodology",
        "Process and methodology questions for engineering projects and management.",
        "Car race app demonstration, Reddit posting protocol.",
        ["134", "968"],
    ),
    (
        "Domain registration & web hosting",
        "Domain name registration, hosting, mail account migration.",
        "Domain nic.sm, migrate email cPanel non-control-panel, MOSS 2007 FIPS, Azure vs GAE vs AWS.",
        ["952", "320", "2446"],
    ),
    (
        "Locksmithing / physical security / picking",
        "Physical lockpicking and physical security.",
        "Pick lock closed for inventory.",
        ["1357"],
    ),
    (
        "Photoshop / image editing software",
        "Image-editing software workflows (Photoshop, PDFs).",
        "Photoshop script text from user, select layer Photoshop, file size PDFs, sprite squash/stretch, draw images spritesheets, quality printout settings.",
        ["174", "1771", "2931", "4041", "4145", "1047"],
    ),
    (
        "Browsers and OS-side software issues",
        "Browser-side / OS-side application errors and bugs (overlap with OS install but more app-level).",
        "Insert linebreak long string, Permalinks structures, How Do We Keep ... (worldbuilding overlap), config icons WYGWAM.",
        ["246", "1650"],
    ),
    (
        "Startup / business strategy",
        "Startup operations, founders, exec strategy.",
        "Find startup partners platforms.",
        ["1978"],
    ),
]


def main() -> None:
    with SAMPLE_PATH.open(encoding="utf-8") as f:
        sample = json.load(f)
    all_ids = {x["id"] for x in sample}

    # Collect assignments
    assigned: dict[str, str] = {}  # id -> cluster name
    duplicates: list[tuple[str, str, str]] = []  # (id, prev_cluster, new_cluster)

    for name, _desc, _reason, ids in CLUSTERS:
        for tid in ids:
            if tid in assigned:
                duplicates.append((tid, assigned[tid], name))
            else:
                assigned[tid] = name

    unknown = [tid for tid in assigned if tid not in all_ids]
    missing = sorted(all_ids - set(assigned.keys()), key=lambda x: int(x))

    print(f"Total sample: {len(all_ids)}")
    print(f"Assigned (unique): {len(assigned)}")
    print(f"Duplicates: {len(duplicates)}")
    for d in duplicates[:20]:
        print("  dup:", d)
    print(f"Unknown ids (not in sample): {len(unknown)}")
    for u in unknown[:20]:
        print("  unknown:", u)
    print(f"Missing (not assigned): {len(missing)}")
    for m in missing[:80]:
        # Print text to help
        text = next(x["text"] for x in sample if x["id"] == m)
        print(f"  miss {m}: {text}")

    if duplicates or unknown or missing:
        return

    # Build the proposal output
    clusters_out = []
    for name, desc, reason, ids in CLUSTERS:
        clusters_out.append(
            {
                "name": name,
                "description": desc,
                "text_ids": ids,
                "reasoning": reason,
            }
        )

    now = datetime.now()
    ts_str = now.strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:4]
    out_path = WORKSPACE / "proposals" / f"prop_{ts_str}_{short}.json"

    payload = {
        "timestamp": now.isoformat(timespec="seconds"),
        "sample_size": len(sample),
        "sample_strategy": "random",
        "style": "humanities/lifestyle/games lens (proposer #3)",
        "existing_clusters_considered": False,
        "clusters": clusters_out,
        "unclustered_ids": [],
        "observations": (
            "StackExchange post titles span ~70+ topical sites. Strong non-technical "
            "presence: language-learning (Japanese, German, Spanish, Russian/Ukrainian, "
            "Italian, Portuguese, English grammar), religion (Christianity, Judaism, "
            "Islam, Hindu/Vedas, mythology), philosophy, history, parenting, "
            "interpersonal advice, brewing, cooking, photography, woodworking, RPGs, "
            "ham radio, cryptocurrency. Technical share is dominated by SharePoint / "
            "Sitecore / Magento / Joomla CMS administration and PHP/JS/SQL web "
            "development. Math/physics/CS-theory each have small but distinct buckets. "
            "Many short-title posts could plausibly fit multiple clusters; I leaned "
            "toward the dominant topical site rather than the surface technology. No "
            "'misc/other' buckets were used."
        ),
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Cluster count: {len(clusters_out)}")


if __name__ == "__main__":
    main()
