"""
Protected terms list — acronyms, medicine names, and internet slang that must
NEVER be translated. Edit this file to add more terms as you find them in your data.

UNCHANGED from the previous pipeline.
"""

# Mental health acronyms / conditions
ACRONYMS = [
    "ADHD", "PTSD", "CPTSD", "OCD", "BPD", "NPD", "ASPD", "GAD", "SAD",
    "MDD", "BD", "BPI", "BPII", "DID", "PMDD", "PPD", "PMS", "ARFID",
    "CBT", "DBT", "EMDR", "ERP", "ACT", "SSRI", "SNRI", "MAOI", "TCA",
    "NDRI", "AA", "NA", "OD", "SI", "SH", "ED", "EDNOS", "ASD", "OCPD",
    "IBS", "PCOS", "CFS", "ME/CFS", "CRPS", "TBI",
]

# Common Reddit / internet acronyms and slang that should stay in English
NET_SLANG = [
    "tbh", "imo", "imho", "fyi", "tw", "cw", "afaik", "idk", "smh", "ngl",
    "fr", "lol", "lmao", "omg", "wtf", "btw", "irl", "dm", "op", "aita",
    "eli5", "psa", "til", "yolo", "rn", "atm",
]

# Common medicine names (generic + brand) seen in mental health posts
MEDICINES = [
    "Sertraline", "Zoloft", "Fluoxetine", "Prozac", "Escitalopram", "Lexapro",
    "Citalopram", "Celexa", "Paroxetine", "Paxil", "Fluvoxamine", "Luvox",
    "Venlafaxine", "Effexor", "Duloxetine", "Cymbalta", "Desvenlafaxine",
    "Bupropion", "Wellbutrin", "Mirtazapine", "Remeron", "Trazodone",
    "Amitriptyline", "Nortriptyline", "Clomipramine", "Imipramine",
    "Lamotrigine", "Lamictal", "Lithium", "Valproate", "Depakote",
    "Carbamazepine", "Tegretol", "Quetiapine", "Seroquel", "Aripiprazole",
    "Abilify", "Risperidone", "Risperdal", "Olanzapine", "Zyprexa",
    "Ziprasidone", "Geodon", "Lurasidone", "Latuda", "Clozapine", "Clozaril",
    "Haloperidol", "Haldol", "Alprazolam", "Xanax", "Clonazepam", "Klonopin",
    "Lorazepam", "Ativan", "Diazepam", "Valium", "Buspirone", "Buspar",
    "Hydroxyzine", "Vistaril", "Propranolol", "Adderall", "Vyvanse",
    "Ritalin", "Methylphenidate", "Concerta", "Dexedrine", "Strattera",
    "Atomoxetine", "Guanfacine", "Intuniv", "Clonidine", "Modafinil",
    "Gabapentin", "Pregabalin", "Lyrica", "Naltrexone", "Suboxone",
    "Methadone", "Melatonin",
    # Addiction / substance-use related (r/addiction, r/opiates etc.)
    "Tramadol", "Oxycodone", "Oxycontin", "Hydrocodone", "Vicodin",
    "Percocet", "Fentanyl", "Heroin", "Codeine", "Morphine", "Kratom",
    "Cocaine", "Meth", "Methamphetamine", "MDMA", "LSD", "Xanax",
    "Benzos", "Benzodiazepines", "Nicotine", "THC", "CBD",
]

ALL_PROTECTED = sorted(set(ACRONYMS + NET_SLANG + MEDICINES), key=len, reverse=True)