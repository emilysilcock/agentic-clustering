#!/usr/bin/env python3
"""Build proposer #2 cluster proposal — fine-grained STEM/technical lens.

Manually-classified 500 StackExchange sample. Assignments are based on careful
reading of titles; for ambiguous items, leans toward the dominant signal.
"""
import json
import sys
from pathlib import Path
from datetime import datetime
import uuid

WS = Path(r"C:/Users/emily/Documents/agentic-clustering/results/clustering/stackexchange/seed=0")
SAMPLE = WS / "sample_proposer2.json"

# Build a lookup id -> text for sanity
sample = json.load(open(SAMPLE, encoding="utf-8"))
id2text = {r["id"]: r["text"] for r in sample}

# Each cluster: name, description, list of text_ids
clusters = []

def C(name, description, ids, reasoning=""):
    # de-duplicate and sanity check
    seen = set()
    out = []
    for i in ids:
        if i in seen:
            raise ValueError(f"dup {i} in {name}")
        seen.add(i)
        if i not in id2text:
            raise ValueError(f"unknown id {i} in {name}")
        out.append(i)
    clusters.append({
        "name": name,
        "description": description,
        "text_ids": out,
        "reasoning": reasoning,
    })

# --- TECHNICAL / STEM clusters --------------------------------------------

C("Python programming",
  "Questions about Python language, libraries (pandas, numpy, cvxpy, docplex), scripts, and runtime errors.",
  ["3613", "2046", "1470", "933"],
  "Python-specific identifiers (cvxpy, docplex Python module, importing arrays, Python forward-testing).")

C("JavaScript / Node / front-end web JS",
  "Client-side JavaScript, Node.js, Angular, d3.js, and related front-end scripting questions.",
  ["27", "1853", "1397", "3011"],
  "Angular+Node+MySQL, d3.js async cancel, JS script vars, JS geolocation TypeMismatchError.")

C("PHP / web app development (general PHP)",
  "PHP server-side scripting questions not tied to a specific CMS.",
  ["1998", "2109"],
  "Spanish PHP+mysql duplicate-insert ordering question; open-source PHP webchat app.")

C("Salesforce / Apex / Visualforce / SOQL",
  "Salesforce platform development: Apex, SOQL, Visualforce, formula fields, portals, WSDL.",
  ["643", "3917", "3188", "2488", "2400", "2357"],
  "Generate Enterprise WSDL, Portal commission sharing, apex:outputField sObject, formula buttons, "
  "Low Nice Date, 'Name (with presence)' field — classic Salesforce/SFDC terms.")

C("Sitecore CMS",
  "Questions about the Sitecore .NET CMS — indexing, SXA, search, forms.",
  ["714", "331", "604", "1900"],
  "SXA components, Sitecore indexing commit policy, WFFM (Web Forms For Marketers), Sitecore 9 Azure Search.")

C("WordPress",
  "Theme, plugin, and admin questions for the WordPress CMS.",
  ["563", "3243"],
  "Postmash plugin, TwentyTen theme menus.")

C("Magento e-commerce platform",
  "Frontend/backend customisation and admin issues for Magento.",
  ["1569", "2190", "2831", "1943"],
  "Magento phtml template loading, configurable product cloning, Product Attributes Index, "
  "Magento layout/block controllers.")

C("Joomla CMS",
  "Joomla template / admin questions.",
  ["3815"],
  "Joomla 3 Protostar template disappeared.")

C("Drupal / shopping cart / Mamp local dev",
  "Miscellaneous PHP-CMS-adjacent: shopping cart, local LAMP/MAMP setup, .htaccess.",
  ["2730", "1249"],
  "Shopping cart script recommendation; Mamp Pro .htaccess for index.php removal.")

C("SharePoint admin & development",
  "SharePoint 2007/2010 customisation, FBA, ScopeDisplayGroup, PowerShell, custom WCF.",
  ["2139", "2298", "1948", "3084", "3892"],
  "FBA credentials to WCF on SharePoint, ScopeDisplayGroup PowerShell, SP2010 sites, "
  "SystemUpdate event receiver, programmatic SP2007 vs 2010 detection.")

C("SQL databases and database admin",
  "SQL Server, MSSMS, Sybase/SQL Anywhere, SSAS, query/connection issues, table existence.",
  ["166", "2253", "2873", "1652", "3085", "3736", "447"],
  "Typecast string as id, SQL Management Studio slow connect, SSAS for tick/bar data, "
  "SQL Anywhere 9 networking, Russian 'check if table exists in DB', per-component DBs, "
  "transactions with per-query connections.")

C("Linux / Unix sysadmin & shell tooling",
  "Linux internals (/proc), vim, image file management on Linux.",
  ["1060", "2287", "2714"],
  "vim yank without visual block; /proc/net/dev; Linux image folder organizer.")

C("Windows desktop & client admin",
  "Windows 7/10/XP client issues — updates, dual-boot, Outlook indexing, Outlook export.",
  ["2769", "2223", "3465", "1381", "3326", "3999"],
  "Windows 10 updates, manually start Windows Update, Outlook 2007 re-index on XP, "
  "Windows 7 dual-boot from VHD, indentation outliner for XP, exporting Outlook emails.")

C("Server administration & Windows Server services",
  "Server reconnection, Active Directory authentication, Certificate Services, thin clients.",
  ["78", "737", "3434", "1654"],
  "Reconnect server safely between farms, AD auth in hosted env, Certificate Services cleanup, "
  "Acer WT300 thin client password.")

C("Networking, routing, switches, VLANs",
  "Network troubleshooting: subnets, multicast, VLANs, QoS, ping issues.",
  ["1437", "1808", "1993", "2624", "3757", "2150"],
  "New IP on subnet change, multicast across VLANs on Juniper, ping across subnets, "
  "block VLANs between Juniper switches, QoS classification, wifi devices in range.")

C("IT security & cryptography & malware",
  "Application/network security, malware/ransomware, key storage, virus scanning.",
  ["3143", "1090", "2505", "1574", "3145", "1568", "2241", "45", "3723", "3034", "4019", "3709"],
  "JPEG/EXIF shell upload, redsocks->Tor, mobile eavesdropping, Zigbee key transport, "
  "Tor ExitNodes file, Tor circuit compromise, virus scanner URL inspection, Windows cert store key, "
  "Gmail hacked, ransomware->Tor, Flash updates, biggest unsolved IT security problems.")

C("Cryptocurrency / Monero / Bitcoin / Ethereum / blockchain",
  "Mining, wallets, contracts, blockchain mechanics across Monero, Bitcoin, Ethereum.",
  ["1555", "2742", "2233", "3134", "1206", "1695", "1101", "3849", "3948", "575", "2581"],
  "Smart contract deletion, BTC vs Monero mining pool, Monero denominations, 4-PC mining, "
  "Monero in e-commerce, account history view, Trinity wallet keychain, Monero blocks/min, "
  "Solidity keccak256, Truffle compile error, block.timestamp safety.")

C("Raspberry Pi / Arduino / embedded hardware electronics",
  "Single-board computers, GPIO, sensors, low-power microcontrollers.",
  ["986", "1120", "245", "1560", "595", "3453"],
  "RPi SSH refused, GPIO HIGH/LOW conditional, USB short-circuited Pi, micro pump on RPi, "
  "ADC with Sharp IR sensor, RPi underpower detection.")

C("PC hardware, peripherals, monitors & displays",
  "Desktop PCs, GPUs, monitors, DP/DVI adapters, scanners, keyboards.",
  ["35", "172", "3364", "3469", "3103", "3448", "1551", "578"],
  "How a scanner works, hardware-only remote desktop, KVM-over-IP solutions, dual GPUs/monitors, "
  "wide-gamut monitor calibration, DP-to-DVI adapter active/passive, caseless dust-free PC, "
  "PC audio to phone mic.")

C("Mobile / smartphone hardware & OS issues",
  "Mobile device hardware/peripheral installation, brightness, custom keyboards across iOS/Android/Symbian.",
  ["588", "1119", "3223"],
  "Samsung NC10 brightness dimming, .sis install on Mac, iOS+Android custom keyboard SDK.")

C("Android development & UX",
  "Android-specific UI/UX dev questions.",
  ["491", "468"],
  "Android up/down arrows in ExpandableListView, JSF backend + Android client architecture.")

C("Web/UI design patterns & UX",
  "User experience and interface design choices: layout, form widths, RSS icons, mobile feedback, modal close buttons.",
  ["922", "1938", "812", "788", "3125", "1516", "4151", "1946", "3750", "509", "2001", "706"],
  "Sent-message-box UX, maintenance category graphic, RSS icon placement, close button side, "
  "mobile input feedback, present 2 buttons on tablet, input controls same width, right-align Help menu, "
  "model photo conversion rate, transit bus stop buttons, download size affecting users, ad network payout UX.")

C("Web development misc (HTML/CSS/images/redirects/regex)",
  "Front-end web utilities: HTML tag parsing, CSS selectors, redirects, regex, image upload, alt video players.",
  ["962", "994", "2866", "2215", "23", "211", "1745", "1559", "3724"],
  "HTML tag multi-line detection, 30x redirect codes, regex first/last name, cssSelector child, "
  "save image without overriding, animated GIF background removal, combine PNG icons, alt video player, cPanel redirect loop.")

C("Reverse engineering & low-level binary debugging",
  "Disassembly, IDA, objdump, GDB, stack inspection.",
  ["2877", "2509", "534", "1376", "903"],
  "idapython segments, objdump MIPS, GDB MI vs stdout, first 16 stack bytes, "
  "binary representation utility.")

C("Mathematica / Wolfram / symbolic computation",
  "Wolfram Mathematica notebooks, custom GUI controls, MatrixPlot, Do/Nest.",
  ["463", "368", "3070", "1218"],
  "Two color functions in MatrixPlot, Replace Do loop with Nest, Mathematica GUI controls, "
  "forcing graph not to resize (Mathematica).")

C("Statistics, regression, time series, experimental design",
  "Applied statistics: regression models, ARCH/GARCH, Kaplan-Meier, repeated measures, Wald test, slope comparison.",
  ["2290", "1888", "2206", "3591", "2370", "3729", "3577", "3644", "503"],
  "Logit/probit multi-response, repeated measures non-temporal, Chi-sq vs CI Wald, "
  "Cumulative Incidence vs Kaplan-Meier, ARCH/GARCH innovation, time series SAS vs R, "
  "compare slopes between regressions, winsorization & standardization, L-infinity regression solver.")

C("Probability theory & stochastic processes",
  "Pure probability, martingales, denoising covariance, probability puzzles, electronic payment probability.",
  ["92", "2070", "871", "3676"],
  "Equivalent martingale measures, n points with given mean/covariance, electronic payment probability, "
  "min-money guess-the-number worst case.")

C("Calculus, real analysis, differential equations & numerical methods",
  "Integrals, gradient descent, Helmholtz decomposition, smoothness indicators, fixed-point iteration history.",
  ["151", "1921", "2352", "1406", "3870", "2331"],
  "Gradient descent convergence, double integral area vs volume, Fourier series usefulness, "
  "Helmholtz decomposition, WENO smoothness indicator, fixed-point iteration history.")

C("Linear algebra & matrices",
  "Matrix structures, determinants, abstract linear algebra applications.",
  ["2610", "2775", "4023"],
  "Determinant of polynomial matrix, row-major vs column-major, abstract LA for engineers.")

C("Discrete math, graph theory, combinatorics, number theory",
  "Graphs of small order, binomial coefficients, irrational classifications, Diophantine equations.",
  ["3307", "969", "3861", "3597"],
  "5-vertex graphs degree 2, n-choose-k in Catalan, Theaetetus quadratic irrationals, x^6+y^10=z^15.")

C("Geometry & algebraic geometry",
  "Geometry (high school and beyond), isosceles trapezoids, Kahler differentials of affine varieties.",
  ["2868", "2540", "3731"],
  "What is high-school geometry today, construct isosceles trapezoid, Kahler differentials.")

C("Mathematics education & pedagogy",
  "Teaching math, motivating students, references.",
  ["2507"],
  "Artistic works with mathematical aspects (math-education-leaning).")

C("Physics — quantum mechanics & quantum information",
  "Quantum circuits, density operators, teleportation, quantum-information primitives.",
  ["2605", "1352", "1659"],
  "Greater-than quantum circuit, density operators & separable states, quantum teleportation second classical bit.")

C("Physics — classical mechanics, EM, thermodynamics & relativity",
  "Center of mass, forcefield medium, special relativity 4-volume, electrical safety, AM IF frequency, steel structure temperature.",
  ["3799", "3003", "1703", "4101", "3096", "1422"],
  "Center of mass of solid, magnetic-field medium, 4-volume in SR, safe human voltage, "
  "AM 455 kHz IF, steel altering temperature.")

C("Physics — astrophysics, cosmology & space",
  "Black hole information, space suit puncture, Soyuz landing, SpaceX landings, LEO-to-Mars cost.",
  ["3429", "1997", "3400", "2122", "1475"],
  "Evaporating BH info, EVA suit puncture, Soyuz water landing, SpaceX Falcon land, kg to Mars LEO cost.")

C("Chemistry & food/drink chemistry (homebrewing, fermentation, beer/mead)",
  "Brewing off-flavours, clarification, mead step-feed, ginger in mead, chamomile addition.",
  ["2738", "3933", "750", "931", "2649"],
  "Off-flavours homebrew, clearer home-brewed beers, step-feeding mead, ginger amount in mead, chamomile technique.")

C("Biology — human medical, anatomy & physiology",
  "Cancer vs abnormal cells, BMP factors, gas pain, common cold, immune, antidepressant timing, deltoid cracking, CPR statistics, vegan->meat.",
  ["1999", "127", "2276", "1526", "582", "3835", "369", "3925"],
  "Cancerous vs abnormal cells, BMP-12/13/14 in tendons, why gas hurts, common cold immune effect, "
  "antidepressant vs coffee timing, deltoid cracks on lift, CPR for lightning strike, vegetarian->meat steps.")

C("Animal biology, pet care & veterinary",
  "Pets (hamsters, dogs, cats, snakes), poultry hypnosis, fish/dolphin/shark, ornithology, flea medicine.",
  ["362", "1908", "1847", "1854", "2317", "598", "2832", "4028", "2604", "3622"],
  "Djungarian hamster, flea medicine human use, chemical castration dog behavior, hypnotized chickens, "
  "snake shedding, springer-spaniel calm, shark size by fin, medieval ornithology, cat movement responsiveness, "
  "treats after a fall (dog/pet training).")

C("Botany, gardening & plants",
  "Plant identification, bonsai care, quince edibility, ginkgo leaves, purple daisy.",
  ["295", "1570", "1585"],
  "Ginkgo bonsai browning, quince safe to eat, purple daisy plant ID and hardiness zones.")

C("Cooking & food preparation",
  "Saving leftovers, take-away fries crispness, food safety, kettle plastic, food techniques.",
  ["614", "637", "4121"],
  "Saving uneaten dinner, fries crispy longer, plastic kettle boiling.")

C("Mathematics — applied, optimization & math software",
  "L-inf regression, soft constraints, SPH sim parameters, ParaView plotting.",
  ["2654", "2407"],
  "SPH simulation parameters, ParaView 2D plot from data sets.")

C("Engineering & signal processing — control, signals, mechanics",
  "Digital controller design, transfer functions, mass estimation, robotic vision, Rankine/Coulomb earth pressure, civil/geotech.",
  ["157", "322", "3334", "3381"],
  "Digital controller variable sample time, mass from transfer function/bodeplot, visual servoing, "
  "earth-pressure theories applications.")

C("GIS / geospatial / cartography",
  "QGIS, basemaps, geospatial open data, soil data, geolocation queries on places.",
  ["1802", "1954", "1969", "3739", "558"],
  "UK places by radius, Rutgers geospatial, soil depth WA, QGIS basemaps, GIS open-data funding.")

# --- LINGUISTICS & HUMAN LANGUAGE clusters --------------------------------

C("English language usage & grammar",
  "English usage, tense, comparative forms, punctuation, idioms, prepositions, slang interpretation.",
  ["1190", "1832", "3604", "1146", "499", "636", "3941", "3689"],
  "quicker vs more quickly, 'would have been happy' tense, 'Email me' vs 'mail to me', "
  "stand by/at the bar, pronouncing two-consonant syllable, pronouncing FAQ, joining pro-sentence punctuation, "
  "translation of 'unpublish'.")

C("German language",
  "Deutsche Sprache, German usage and grammar.",
  ["1801", "3818", "1668", "1095", "1813", "3108"],
  "dahingehend vs derartig, Numerus des Praedikats, Zertifikat Deutsch, 'smart device' auf Deutsch, "
  "'Universitaet zu Koeln' usage, Heidegger zeug vs art (Phil/German term overlap — leans German).")

C("Spanish language",
  "Spanish usage, vocabulary, regional Spanish.",
  ["3580", "3817", "924", "3101"],
  "fue vs era, 'feria' as money, blueberries/cranberries in Spanish, despues vs luego.")

C("French language",
  "French grammar, idioms, regional French, Canadian French corpora.",
  ["845", "1624", "2411", "4141", "3707", "3848"],
  "« (Pas) pour cinq cennes », petites chambres article placement, French corpora, "
  "accord participe passe, French audio CC-licensed, French Canadian record dictionary.")

C("Russian / Ukrainian / Slavic languages",
  "Russian, Ukrainian usage, vocabulary differences, syllabic spelling.",
  ["826", "3524", "2248", "2728", "882", "4108", "409"],
  "над вязаньем/вязанием, вслед за чем vs вслед чему, наречений/жених (UA), Чей/Чья vs Кого, "
  "blending stump Russian word, наголошувати 'тато' UA, finding text 'безоговорочной теории'.")

C("Italian language",
  "Italian usage and vocabulary.",
  ["1259", "1147"],
  "'Di dove sei' vs 'Da dove vieni'; 'cavilloso' meaning.")

C("Japanese language",
  "Japanese grammar, particles, kana/kanji, translation.",
  ["3632", "1087", "3789", "3274", "3340", "3549"],
  "と particle role, の->ん substitution, て-form vs と/や, コレ meaning, 'ki' meanings, "
  "Xがどこに見つけられますか acceptability.")

C("Korean / Chinese / other East-Asian language",
  "Korean expressions; East-Asian L2 questions not covered elsewhere.",
  ["164"],
  "썰렁 / 썰렁해 explanation (Korean).")

C("Latin / classical and ancient languages",
  "Latin / classical languages questions.",
  ["3152"],
  "Latin word for 'respectively'.")

C("Language learning, bilingualism, IPA & language acquisition",
  "L2 learning techniques, IPA, bilingual accents, sign language, baby language exposure, language tools.",
  ["219", "3844", "2102", "1216", "389", "3879", "1629", "1107"],
  "Newborns and language sounds, IPA advantage with phonetic spelling, Spanish+French simultaneous, "
  "bilinguals' accents, Ukrainian alphabet start, 1yr-old foreign-language phrases, "
  "Nicaraguan Sign Language, Spanish+English reading software.")

C("Linguistics — theory, terminology & corpora",
  "Theoretical linguistics: SFL grammatical metaphor, phonotactics, trigram datasets, predicting word knowledge.",
  ["1708", "3077", "332", "1927"],
  "Grammatical metaphor in SFL, phonotactic rules term, English word trigrams dataset, predict-word-knowledge.")

# --- HOBBIES / LIFESTYLE / GAMING -----------------------------------------

C("Tabletop role-playing games (D&D, Pathfinder, 7th Sea, etc.)",
  "Rules questions for tabletop RPGs: classes, spells, dwarven gear, combat, rerolls, NPC immersion.",
  ["37", "884", "341", "170", "1924", "1545", "2556", "1784", "3514", "3392", "3533", "2846"],
  "Anima mage prestige class, potion weight, 7th Sea stat blocks, opposite-characteristic immersion, "
  "skill rerolls, Wood Elf Command Animal, tactical movement engaged, familiar touch spell, "
  "Council of Magic, flanking charge, familiars advance in power, dwarven clothing material.")

C("Video games & game-specific gameplay",
  "Specific video-game gameplay/walkthrough questions (Mass Effect, Starcraft, Portal 2, Two Worlds, King's Bounty, Gangstar Vegas).",
  ["3417", "2386", "3751", "4149", "2236", "1912"],
  "Two Worlds teleporters, Mass Effect Element Zero, Portal Wheatley capture, Starcraft 2 bonus pool, "
  "King's Bounty undead penalty, Gangstar Vegas alt-rock song.")

C("Game development",
  "Game programming and engines (XNA on WP7).",
  ["2145"],
  "Drawing texture line between vectors in XNA WP7.")

C("Chess, puzzles, board & card games",
  "Chess problems, cryptic crosswords, poker odds & strategy.",
  ["2111", "1966", "378", "954", "2672"],
  "Hiding chess puzzle in plain sight, cryptic crossword clue, pot odds, betting vs cards in poker, "
  "who wins the hand.")

C("Sports — general, soccer/football, baseball, badminton, MLB, Australian football",
  "Sports rules and strategy questions across multiple sports.",
  ["307", "2829", "2344", "1974", "3527", "2349", "1618"],
  "Referee added time, stolen base catcher rule, MLB slide rule, badminton opening service, "
  "reduced-numbers Australian Rules, cricket bat dimensions, climbing autoblock+belay (sports-adjacent).")

C("Fitness, exercise, weight training",
  "Workout routines, deadlifts, BJJ, grip on chin-up bar, belly size, cardio efficiency.",
  ["73", "557", "1080", "1191", "1335", "2120", "1298", "2221"],
  "Romanian deadlifts, obstacle race workout, BJJ improvement, chin-up grip, belly after workout, "
  "compound lifting cardio, Starting Strength calories, effective workout question.")

C("Photography — equipment, lenses, technique",
  "Cameras, flashes, lenses, filters, batteries, portraits, sparkly subjects, point-and-shoot vs SLR.",
  ["1416", "2227", "1086", "2345", "49", "3390"],
  "Aperture vs flash power, filter for prime lens dust, improving portraits, flash batteries, "
  "photographing sparkly objects, P&S vs SLR.")

C("Music — theory, instruments, recording & identification",
  "Music gear (pedals), classical copyright, dB mixing, song ID, music theory/arrangements.",
  ["698", "2825", "91", "2468", "1199", "3106", "1426"],
  "Chorus pedal placement, classical excerpt copyright, Beats trailer song, Esperanza Spalding/Zappa arrangement, "
  "Iron and Wine + Jason Isbell openers, 0dB mixing reference, Romeo and Juliet reference in 'Love'.")

C("Movies, TV & fiction analysis",
  "Plot/character/identification questions about movies, TV shows and films.",
  ["644", "2425", "2850", "3365"],
  "Mr. Robot framing, Logan claws bleeding, 'In Time' time source, Mazer Rackham & Graff conversation.")

C("Literature & novels analysis",
  "Literary analysis, novels, books-from-description, Mr. Toad's castle.",
  ["262", "1122", "2575", "3647"],
  "Crime and Punishment chapter 2, captain-of-starship novel, first present-tense novel, Mr. Toad castle location.")

C("Writing craft, screenwriting & narrative techniques",
  "Writing craft topics: Mary Sue trope, line spacing, novels conventions.",
  ["2052", "3601"],
  "Mary Sue concept; line spacing in novels.")

C("Religion — Judaism",
  "Jewish religious practice, study, halakhah, teshuva, Hasidic stories, scripture interpretation.",
  ["2074", "3043", "2572", "3320", "624"],
  "Havruta solo study, teshuva book, Hasidic rabbi with no beard, Islamic marital law misclassed? "
  "(actually Islam — moved), Jewish cemeteries list. 3320 actually goes to Islam — moving.")

# Fix: 3320 belongs in Islam, not Judaism. Rebuild the two.
# Edit clusters list:
clusters[-1]["text_ids"] = [i for i in clusters[-1]["text_ids"] if i != "3320"]
clusters[-1]["reasoning"] = "Havruta study, teshuva book, Hasidic rabbi with no beard, list of Jewish cemeteries."

C("Religion — Christianity & biblical interpretation",
  "Bible interpretation, OT/NT, Greek vs Hebrew texts, scripture references.",
  ["3737", "230", "1869", "3507", "959", "1447"],
  "Leviticus 20:20 meaning, Isaiah 30:20-21 Greek vs Hebrew, Leviticus 8 ordination vs Matt 5:17, "
  "tennis-fallacy reframed? — actually 3507 is philosophy. Move. Reassigning below.")

# 3507 belongs in philosophy (fallacy). Fix:
clusters[-1]["text_ids"] = [i for i in clusters[-1]["text_ids"] if i != "3507"]
clusters[-1]["reasoning"] = "Leviticus 20:20 meaning, Isaiah 30:20-21 Greek vs Hebrew, Leviticus 8 'ordination' & Matt 5:17, scriptures Timothy knew, Jehoahaz/Jehoash overlap."

C("Religion — Islam",
  "Quranic interpretation, Islamic law, prayer.",
  ["3320", "843"],
  "Marital intercourse closed-doors hadith, Quran punctuation.")

C("Religion — Hinduism / Indian religion",
  "Hinduism, Indian myth, mantra identification, Krishna/Indra stories.",
  ["4003", "214", "305"],
  "Mantra in video, Satya Sai names of Arjuna, menstruation after Indra's sin.")

C("Religion — comparative / mythology / non-Abrahamic",
  "Greek myth, comparative religion, soul/sleep metaphysics, spirituality.",
  ["3112", "3635"],
  "Helios myth question, does soul go elsewhere in deep sleep.")

C("Philosophy & logic",
  "Logic puzzles, philosophical fallacies, mind/visualisation, abstract philosophy.",
  ["1952", "3278", "2028", "3507"],
  "Is logic logic because God said so, female-redhead syllogism fallacy, visualization & intuition, "
  "Federer tennis fallacy.")

C("History — ancient, medieval, modern",
  "Historical questions: Charlemagne & Vikings, Hasmoneans/Jehoash overlap (moved out), passport archives, mariners.",
  ["3301", "1805", "3575", "1130"],
  "Charlemagne -> Vikings, Hamburg-Chile mariners, Polish/Hungarian passport archives, "
  "unregistered death after 1900 in US.")

C("Genealogy & family history",
  "Family-history tools and records research.",
  ["465"],
  "Gramps to family-history book.")

C("Travel — visas, transit, customs, accommodation",
  "Country-specific travel, visas, transit, accommodation, customs duty, ferry, expat tax.",
  ["3290", "1669", "775", "913", "3262", "2585", "1457", "3476", "1721", "2060", "3888", "1540", "2274"],
  "Argentina electronics duty, EU sibling dependent evidence, non-EU partner working in DE, "
  "Melbourne->Guangzhou, UK road trip US, Mexico->Hawaii bags, London->Luton time, "
  "Ireland short-stay with US job, India->UAE money, US immigration overstay, US-paid worker living UK on dependent visa, "
  "H1B after termination, accommodation in Dandong China.")

C("Personal finance, taxes & investments",
  "Personal income tax, savings, expat tax, investment accounts, electronic payments general.",
  ["123", "988", "90", "456", "993", "490"],
  "Expense-savings ratio, freelance + full-time tax, long-term savings for daughter, "
  "frequent-flyer aggregator, PPACA fee for expats, moving expenses deduction.")

C("Quantitative finance / trading",
  "Derivatives stress tests, dual-settlement backtest, electronic-payment probability.",
  ["776", "1032"],
  "FX gamma stress test, dual settlement market backtesting.")

C("Cars, motorcycles, mechanics & automotive maintenance",
  "Vehicle troubleshooting, oil change, brake fluid, MOT failure, Camaro starter, octane effects, turbo detection, clicking engine.",
  ["1175", "2666", "218", "2513", "345", "2504", "687", "2607", "316"],
  "VW Golf timing, VW TDI oil change, DOT 4 vs 4+, Ibiza Mk2 MOT emissions, '67 Camaro hot starter, "
  "sub-optimal octane efficiency, turbocharger detection, clicking on start, coolant label meaning.")

C("Outdoors — hunting, fishing, climbing, gold panning",
  "Outdoor activities — hunting jargon, gold panning signs.",
  ["4043", "3858"],
  "'glass' in hunting, water-edge signs for gold panning.")

C("Parenting & childcare",
  "Babies/children's health, sleep, feeding, breastfeeding, MMR side effects.",
  ["641", "2677", "3716"],
  "Baby not eating after MMR, breastfeeding older while newborn, 1yo not sleeping.")

C("Mental health, life advice & social-emotional",
  "Romance/job dilemmas, sharing work confidence, friendships, weight, memory tips.",
  ["780", "1524", "227", "42", "1647", "1767", "2871"],
  "Bow out of commitment leaving job, improve short-term memory, gain weight naturally, "
  "tell employer I'm in love, signal friendship not romance, hard to share work, decline surprise birthday.")

C("Workplace & job search advice",
  "Resumes, performance, interviewing, team management.",
  ["1480", "87", "329"],
  "Entry-level experience expectations, measuring team member performance, interview standards for IT dept.")

C("Project & product management / Scrum / Agile",
  "Agile/Scrum/PM topics: roadmaps, story points, sprints, defects, quality.",
  ["1534", "1720", "797", "3957", "3992", "3509", "2713", "3452", "3027"],
  "MVP/roadmap relationship, quality culture, backlog story points, PM split with tech lead, "
  "billing for defects/unplanned, quality of 'design by doing', Hawthorne effect and sprint, "
  "define who works on project, merging migration in API.")

C("Software architecture, build & devtools",
  "Build/test/distribute concerns: multi-project orgs, Jmeter, Selenium, simulating distributed systems, ParaView->XNA build issues, license defaults, end-user docs.",
  ["2701", "1739", "1201", "1584", "1271", "1764", "77", "1926", "1291", "3029"],
  "Multiple WebProjects with shared logic, Jmeter Login before each call, Selenium 'cannot instantiate', "
  "test strategy for distributed nondeterminism, 'Invalid block type' diagnosis, default software license, "
  "end-user doc references, runtime error on 1.1 methods, local server timezones harmful, shared test-device pool.")

C("Legal — patents, IP & copyright",
  "Patents (provisional, prior art, China copying), patent claims, copyright/CC, software licenses.",
  ["1240", "2939", "461", "3164", "2614", "1865", "3061", "3904"],
  "Provisional patent water-tight, patent teaching method, China copying US patent, copyfree 'crayon' license, "
  "largest number of patent claims, prior art request, abandoned provisional reapply, 'how can this be patented'.")

C("Legal — non-patent law / regulations & compliance",
  "GDPR, search warrants, civil evidence, voter registration, third-party info, MTA finances, expat health.",
  ["2497", "1944", "223", "4061", "3771", "1263", "2320"],
  "GDPR developer vs user company, search warrant 3rd party, written civil case statements, "
  "third-party info requests, voter registration citizenship checks, MTA finance accuracy, "
  "driver license data.")

C("Politics, government & public policy",
  "Government, parliamentary term limits, Trump/PACHA, Data.gov APIs.",
  ["3468", "4095", "3814"],
  "Parliamentary term limits, Trump fired PACHA, US agencies <-> Data.gov APIs.")

C("Crafts, art, drawing & design (manual/visual)",
  "Visual art techniques, Illustrator screens, InDesign billboard, design quality, color channel counting.",
  ["1158", "3717", "2174", "1005", "1919"],
  "Polyed paintbrush procrastination, screens of black for grey in Illustrator, billboard > InDesign size, "
  "counting elements in color channel, non-destructive edits.")

C("Personal interest — radio/ham, electronics regulation",
  "Amateur radio regulations, sending coords via VHF.",
  ["2653", "3076"],
  "Amateur radio reception/archive regulation, VHF coords to computers.")

C("Software & web — niche utilities (RSS, Animoto, Outlook, Chrome ext)",
  "Specific consumer/end-user software utility recommendations: RSS readers, video makers, wiki, frequent-flyer aggregator, Chrome ext, Mac PDF tools.",
  ["2936", "987", "4004", "3352", "3566", "2177", "812", "1449"],
  "RSS on iPod Classic, Animoto alternative, inline AJAX wiki, Chrome video downloader, "
  "Mac PDF fuzzy search, offline Google Apps, RSS icon placement (revisit — already in UX), "
  "Tweetdeck for Reddit/SE.")

# 812 is duplicated with UX above; remove from here.
clusters[-1]["text_ids"] = [i for i in clusters[-1]["text_ids"] if i != "812"]

C("Forums, community & moderation tooling",
  "Forum software, moderation, community tools, StackTagz, scheduling/queues.",
  ["3822", "2656", "3612"],
  "Forum mgmt against barbarians, StackTagz topic tracker, shop-floor task queue desktop sim.")

C("Cybersecurity vulnerabilities / pentest exotic topics",
  "Reverse-engineering & malware-related issues touching shell uploads, payment encryption methodology.",
  [],
  "(deferred — overlap with IT security; left empty)")

# remove the empty cluster
clusters = [c for c in clusters if c["text_ids"]]

C("Industrial / process engineering & scheduling",
  "Project scheduling, contract modifications, operations.",
  ["2770"],
  "Add new activities from contract modifications into schedule.")

C("Symbolic regression & philosophy of ML",
  "Philosophical questions about ML methodology — symbolic regression Popperian etc.",
  ["2202"],
  "Symbolic regression Popperian or inductivist (sits on stats/CS-phil border).")

C("Open-data / public datasets",
  "Open-data discovery — soil, mariners, geospatial, projectile coords, FDA, drivers.",
  ["3634", "216"],
  "Sports projectile coordinate datasets; Open FDA adverse reactions schema.")

C("Photography & graphic design publishing — printing & PDFs",
  "Print/PDF output: print list to PDF, billboard size.",
  ["489"],
  "Print list info to PDF.")

C("Mobile dev — iOS / WebView misc",
  "Browsers, user-agent, mobile-specific browser detection.",
  ["1510"],
  "Browser agent string com.google.GooglePlus.")

C("LaTeX / TeX / typesetting",
  "TeXLive, dvi/ps/pdf editor compilation.",
  ["3784", "705"],
  "TeXLive 2011 upgrade notes, editor with code folding and dvi-ps-pdf compile.")

C("Pets / domestic animals / aquaria",
  "Pet rules & cohabitation (some overlap with vet).",
  [],
  "(left empty — domestic-pet items folded into 'Animal biology, pet care & veterinary'.)")
clusters = [c for c in clusters if c["text_ids"]]

# Fashion / clothing / general consumer
C("Clothing & consumer goods",
  "Clothing material choices (cotton socks).",
  ["708"],
  "When cotton socks are actually better.")

# Lifestyle: paranormal / alternative
C("Alternative medicine & wellness claims",
  "Chromotherapy and similar wellness claims, diuretics, ginkgo.",
  ["2703", "909"],
  "Chromotherapy literature support, too many natural diuretics.")

# Misc gaming dev/IT/cards
C("Card games strategy",
  "Poker strategy questions (subset of card-game cluster, plus general).",
  [],
  "(folded into Chess/puzzles/board/card-games cluster.)")
clusters = [c for c in clusters if c["text_ids"]]

# ---- second pass: assign remaining 34 unassigned ids ----

# Add to existing clusters by name
def add_to(name, ids):
    for c in clusters:
        if c["name"] == name:
            c["text_ids"].extend(ids)
            return
    raise ValueError(f"no cluster {name}")

# Business / enterprise software conceptual
C("Enterprise software & business concepts",
  "Vendor-neutral business software topics: BPM software, business process management.",
  ["10"],
  "What is Business Process Management software.")

# Riddles and word puzzles (separate from chess crosswords)
C("Riddles, wordplay & verse puzzles",
  "Cryptic prose riddles and clue-style wordplay.",
  ["198", "338", "1158", "3907"],
  "'A long day for the unravelling Salesman', 'I Can Only Live Where There is Light', "
  "'I polyed my paintbrush then procrastination left me in a predicament', "
  "'Are there eighteen or twenty bars in my castle?'")
# remove 1158 from Crafts cluster (we moved it to Riddles)
for c in clusters:
    if c["name"].startswith("Crafts, art, drawing"):
        c["text_ids"] = [i for i in c["text_ids"] if i != "1158"]

# C / C++ / low-level programming language
C("C / C++ programming",
  "C and C++ language coding tasks, including in Russian-language SO clones.",
  ["205", "841", "3289", "3975"],
  "Russian C++ I/O flags, Russian array indexing, passing arrays to DLL (C/C++), PageRank loop in C.")

# Java / JVM specific
C("Java / JVM languages",
  "Java/Hibernate runtime errors and Java ecosystem.",
  ["220", "2717"],
  "FileNotFoundException with iReport (Java reporting tool); Hibernate ResourceClosedException.")

# ASP.NET / C#
C("ASP.NET / C# web stack",
  "Microsoft .NET ASP.NET MVC, C#, controller/model.",
  ["1489"],
  "Spanish ASP.NET MVC C# controller->model question.")

# Portuguese (Brazilian) language / dev forum
C("Portuguese language & Brazilian dev",
  "Portuguese-language StackExchange (pt.SO) questions.",
  ["3537"],
  "'Forcar a declaracao de propriedades' — Portuguese property-declaration question.")

# Web scraping / data extraction
C("Web scraping & data extraction tools",
  "Tools and methods to extract structured data from web pages.",
  ["1244"],
  "import.io failed — looking for other tools.")

# Optics & physics gadgets / DIY engineering
C("DIY engineering & light-handling devices",
  "Hands-on engineering: making a strong magnifying glass.",
  ["319"],
  "Make a very strong magnifying glass.")

# Retro / vintage computing
C("Retro / vintage computing",
  "Retro home computers and audio chips.",
  ["397"],
  "C64 SID chip output — 8-bit sound.")

# Linux / disk-level recovery
add_to("Linux / Unix sysadmin & shell tooling", ["519"])

# Gmail behaviour / web mail clients
C("Webmail & online account behavior",
  "Behavior of web-based mail accounts (Gmail logins, XMPP at gmx.net).",
  ["602", "902"],
  "XMPP at gmx.net, Gmail last-login display.")

# Conversational greetings as language (folded into language learning)
add_to("English language usage & grammar", ["712", "1215"])

# Russian web dev (form submit to multiple sites)
add_to("PHP / web app development (general PHP)", ["1901"])

# Conspiracy / argumentation
C("Conspiracy & extraordinary claims",
  "Skeptic-style claims about conspiracies, alternative explanations.",
  ["579"],
  "How to refute WTC argument.")

# Women travelling / cultural rules (Islam — rules for men/women travelling alone)
add_to("Religion — Islam", ["1994"])

# Construction / building materials
C("Construction & building materials",
  "Materials selection and construction techniques.",
  ["1996"],
  "Cardboard bales as building material.")

# Income-tax-free territories — finance-adjacent
add_to("Personal finance, taxes & investments", ["2163"])

# Worldbuilding (snowy civilizations)
C("Worldbuilding & speculative fiction",
  "Speculative fiction worldbuilding (climates, cultures).",
  ["3038"],
  "Civilization in cold permanently snowy climate eats what.")

# Energy / power consumption of online video — utility consumer
C("Energy & power consumption (consumer)",
  "Power use of consumer electronics, including video resolution effects.",
  ["3091"],
  "Resolution effect on online-video power consumption.")

# Names / naming
C("Names & naming conventions",
  "Personal-name discussion: 'Sparrell as first/middle'.",
  ["3222"],
  "Sparrell as a first and middle name.")

# Education — teaching practice / pedagogy (non-math)
C("Education & teaching practice (general)",
  "Schools, teaching practice, certifications.",
  ["359"],
  "How to choose the school for practising teaching.")

# Money & cash safety
add_to("Personal finance, taxes & investments", ["387"])

# Long-term storage / consumer questions (mix of preservation)
add_to("PC hardware, peripherals, monitors & displays", ["3401"])

# Windows memory tool
add_to("Windows desktop & client admin", ["3500"])

# Sitecore field/relationship — actually Salesforce/Sitecore? "Relationship field & variables outside of product form" — Salesforce
add_to("Salesforce / Apex / Visualforce / SOQL", ["3702"])

# Admin Console custom plugin permission — could be Magento/Joomla/SP. Generic CMS admin -> SharePoint
add_to("SharePoint admin & development", ["2989"])

# Remove the placeholder card-games entry already filtered



# Travel-by-rail / transit-only items already in Travel
# Insurance / health insurance covered in Personal finance.

# ---- consolidation pass: merge tiny related clusters to land in 60-100 band ----

def merge(into_name, from_names, new_name=None, new_desc=None):
    """Merge from_names into into_name (or rename into new_name)."""
    target = None
    for c in clusters:
        if c["name"] == into_name:
            target = c
            break
    if target is None:
        raise ValueError(f"missing target {into_name}")
    for fn in from_names:
        src = None
        for c in clusters:
            if c["name"] == fn:
                src = c
                break
        if src is None:
            raise ValueError(f"missing source {fn}")
        target["text_ids"].extend(src["text_ids"])
        clusters.remove(src)
    if new_name:
        target["name"] = new_name
    if new_desc:
        target["description"] = new_desc

# 1. Merge Joomla, Drupal/cart/Mamp, WordPress into one "Other PHP-CMS" bucket
merge("WordPress", ["Joomla CMS", "Drupal / shopping cart / Mamp local dev"],
      new_name="WordPress / Joomla / other PHP CMS & shopping carts",
      new_desc="WordPress, Joomla, Drupal-adjacent shopping carts and local PHP CMS setup.")

# 2. Merge Retro/vintage computing into PC hardware
merge("PC hardware, peripherals, monitors & displays", ["Retro / vintage computing"],
      new_desc="Desktop/laptop hardware: GPUs, monitors, KVM/remote desktop, scanners, displays, storage, and retro home-computer hardware.")

# 3. Merge Enterprise software (BPM) into Software architecture
merge("Software architecture, build & devtools", ["Enterprise software & business concepts"],
      new_desc="Software architecture/build/test concerns: multi-project orgs, test frameworks, distributed-system testing, end-user docs, enterprise software concepts like BPM.")

# 4. Merge Web scraping into Web development misc
merge("Web development misc (HTML/CSS/images/redirects/regex)", ["Web scraping & data extraction tools"],
      new_desc="Front-end web utilities: HTML/CSS/regex, redirects, image upload, alt video players, and web scraping/data-extraction tools.")

# 5. Merge Mobile dev iOS/WebView into Web development misc (browser agent)
merge("Web development misc (HTML/CSS/images/redirects/regex)", ["Mobile dev — iOS / WebView misc"])

# 6. Merge ASP.NET into .NET-adjacent — actually merge with Java since both are server-side dev,
#    or better: put ASP.NET into Software architecture (no better home). Put with Java -> "Java / .NET enterprise"
merge("Java / JVM languages", ["ASP.NET / C# web stack"],
      new_name="Java / .NET enterprise backend",
      new_desc="JVM (Java, Hibernate, iReport) and .NET (ASP.NET MVC, C#) backend stacks.")

# 7. Merge Worldbuilding & speculative fiction into Writing craft
merge("Writing craft, screenwriting & narrative techniques", ["Worldbuilding & speculative fiction"],
      new_name="Writing craft, worldbuilding & narrative techniques",
      new_desc="Writing craft: tropes (Mary Sue), formatting/line spacing, and worldbuilding/speculative fiction setup.")

# 8. Merge Names & naming into English usage
merge("English language usage & grammar", ["Names & naming conventions"])

# 9. Merge Conspiracy/extraordinary into Philosophy & logic
merge("Philosophy & logic", ["Conspiracy & extraordinary claims"],
      new_desc="Logic, fallacies, philosophical/metaphysical questions, mind/visualisation, debunking extraordinary claims.")

# 10. Merge Education (general) into Workplace & job search advice -> "Education & workplace"
merge("Workplace & job search advice", ["Education & teaching practice (general)"],
      new_name="Workplace, hiring & education-practice advice",
      new_desc="Hiring, interviewing, team performance, and teaching-practice / school-choice questions.")

# 11. Merge Industrial/process engineering & scheduling into Project & product management
merge("Project & product management / Scrum / Agile", ["Industrial / process engineering & scheduling"],
      new_desc="Agile/Scrum/PM topics, roadmaps, sprints, defects, quality, plus contract-modification scheduling.")

# 12. Merge Symbolic regression / phil of ML into Philosophy & logic
merge("Philosophy & logic", ["Symbolic regression & philosophy of ML"])

# 13. Merge DIY engineering & light-handling into Engineering & signal processing
merge("Engineering & signal processing — control, signals, mechanics",
      ["DIY engineering & light-handling devices"],
      new_desc="Applied engineering: digital control, signal/transfer-function analysis, visual servoing, "
               "geotechnical/earth-pressure theories, DIY optical devices.")

# 14. Merge Energy & power consumption (consumer) into PC hardware
merge("PC hardware, peripherals, monitors & displays", ["Energy & power consumption (consumer)"])

# 15. Merge Construction & building materials into Outdoors (no better home) — actually create
#     a "Home & DIY" bucket using Clothing as nucleus.
merge("Clothing & consumer goods", ["Construction & building materials"],
      new_name="Everyday materials & consumer goods (clothing, building materials)",
      new_desc="Material choice for everyday/consumer items: clothing, building materials.")

# 16. Merge Game development into Video games
merge("Video games & game-specific gameplay", ["Game development"],
      new_name="Video games — gameplay & game development",
      new_desc="Specific video-game gameplay/walkthroughs plus game programming/engines (e.g. XNA).")

# 17. Merge Genealogy & family history into History
merge("History — ancient, medieval, modern", ["Genealogy & family history"],
      new_name="History & genealogy",
      new_desc="Historical questions across ancient, medieval, and modern periods, plus genealogy/family-history research and archives.")

# 18. Merge Latin / classical languages into Language learning
merge("Language learning, bilingualism, IPA & language acquisition", ["Latin / classical and ancient languages"])

# 19. Merge Korean / Chinese / other East-Asian into Japanese (rename CJK)
merge("Japanese language", ["Korean / Chinese / other East-Asian language"],
      new_name="Japanese, Korean & other East-Asian languages",
      new_desc="Japanese grammar/particles/translation; Korean expressions; other CJK-area language questions.")

# 20. Merge Math education into Calculus cluster
merge("Calculus, real analysis, differential equations & numerical methods",
      ["Mathematics education & pedagogy"],
      new_desc="Calculus, real analysis, ODEs/PDEs, numerical methods, Helmholtz decomposition, plus math-education applications.")

# 21. Merge Photography & graphic design publishing — printing & PDFs into Crafts, art (visual design)
merge("Crafts, art, drawing & design (manual/visual)",
      ["Photography & graphic design publishing — printing & PDFs"],
      new_desc="Visual art / manual design: Illustrator/InDesign output, color-channel counting, PDF print output, non-destructive edits.")

# 22. Merge Mathematics — applied, optimization & math software into Engineering & signal
merge("Engineering & signal processing — control, signals, mechanics",
      ["Mathematics — applied, optimization & math software"])

# 23. Merge Open-data / public datasets into GIS (datasets often geospatial)
merge("GIS / geospatial / cartography", ["Open-data / public datasets"],
      new_desc="Geospatial data sources, QGIS, basemaps, place-by-radius queries, and other open public-dataset discovery.")

# 24. Merge Personal interest — radio/ham into Networking
merge("Networking, routing, switches, VLANs", ["Personal interest — radio/ham, electronics regulation"],
      new_name="Networking, routing & radio communications",
      new_desc="Network troubleshooting (subnets, multicast, VLANs, QoS, ping, wifi) and amateur radio/VHF coordination regulations.")

# 25. Merge Webmail & online account behavior into IT security/cryptography
merge("Software & web — niche utilities (RSS, Animoto, Outlook, Chrome ext)",
      ["Webmail & online account behavior"],
      new_desc="Consumer-facing software utilities and online-account behavior: RSS readers, video makers, wiki tools, Chrome extensions, Mac search, Gmail account behavior, XMPP.")

# Now collect all assigned ids; everything must be assigned
assigned = set()
dups = []
for c in clusters:
    for tid in c["text_ids"]:
        if tid in assigned:
            dups.append((tid, c["name"]))
        assigned.add(tid)

all_ids = set(id2text.keys())
missing = sorted(all_ids - assigned, key=lambda x: int(x))
extra = sorted(assigned - all_ids)
print("clusters:", len(clusters))
print("assigned:", len(assigned))
print("missing:", len(missing))
print("dups:", dups)
print("extra:", extra)
if missing:
    for m in missing[:80]:
        print(" MISSING", m, "->", id2text[m])

# ---- write final JSON ----
if not missing and not dups and not extra:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:4]
    out_path = WS / "proposals" / f"prop_{ts}_{uid}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    proposal = {
        "timestamp": datetime.now().isoformat(),
        "sample_size": len(sample),
        "sample_strategy": "random",
        "style": "fine_grained_technical",
        "existing_clusters_considered": False,
        "proposer": "proposer_2_fine_grained_stem_technical_lens",
        "clusters": clusters,
        "unclustered_ids": [],
        "observations": (
            "StackExchange sample is extremely diverse — clear signal of dozens of different sites. "
            "Split aggressively along technical lines: programming language (Python, JS, PHP, C/C++, Java/.NET), "
            "specific platforms (Salesforce, Sitecore, SharePoint, Magento, WordPress/Joomla), "
            "sub-disciplines of math (calculus, linear algebra, probability, discrete, geometry), "
            "physics (quantum, classical, astrophysics), and statistics. Also strong signal for human "
            "languages (English, German, Spanish, French, Russian/Slavic, Italian, Japanese/CJK) and "
            "religion sites (Judaism, Christianity, Islam, Hinduism, comparative). Hobby/lifestyle "
            "groups (RPGs, video games, music, photography, fitness, sports, cars) are also distinct. "
            "Target k=121 implies further splitting (e.g. by language within programming, by sub-discipline "
            "within stats); this proposal lands at k=88 to keep groups defensible from 500 texts."
        ),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(proposal, f, indent=2, ensure_ascii=False)
    print(f"WROTE {out_path}")
else:
    print("NOT writing — assignment issues remain.")
