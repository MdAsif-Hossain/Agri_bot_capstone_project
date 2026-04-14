"""
Seed data for the Dialect Knowledge Graph.

Populates the KG with common Bengali agricultural terms, dialect aliases,
and relations (symptom→disease, disease→treatment, etc.).
"""

import logging
from agribot.knowledge_graph.schema import KnowledgeGraph

logger = logging.getLogger(__name__)


def seed_knowledge_graph(kg: KnowledgeGraph) -> None:
    """
    Seed the KG with initial agricultural terms and relations.

    Categories:
    - Crops (ফসল)
    - Diseases (রোগ)
    - Pests (পোকামাকড়)
    - Symptoms (লক্ষণ)
    - Treatments / Chemicals (চিকিৎসা)
    - Fertilizers (সার)
    """
    stats_before = kg.get_stats()
    if stats_before["entities"] > 0:
        logger.info(
            "KG already seeded (%d entities). Skipping.", stats_before["entities"]
        )
        return

    logger.info("Seeding knowledge graph with agricultural data...")

    # =====================================================================
    # CROPS
    # =====================================================================
    rice = kg.add_entity("ধান", "Rice", "crop")
    kg.add_alias(rice, "ধান", "standard")
    kg.add_alias(rice, "চাল", "standard")  # husked rice
    kg.add_alias(rice, "ভাত", "standard")  # cooked rice (colloquial)
    kg.add_alias(rice, "rice", "english")
    kg.add_alias(rice, "paddy", "english")

    wheat = kg.add_entity("গম", "Wheat", "crop")
    kg.add_alias(wheat, "গম", "standard")
    kg.add_alias(wheat, "wheat", "english")

    jute = kg.add_entity("পাট", "Jute", "crop")
    kg.add_alias(jute, "পাট", "standard")
    kg.add_alias(jute, "jute", "english")

    potato = kg.add_entity("আলু", "Potato", "crop")
    kg.add_alias(potato, "আলু", "standard")
    kg.add_alias(potato, "potato", "english")

    mango = kg.add_entity("আম", "Mango", "crop")
    kg.add_alias(mango, "আম", "standard")
    kg.add_alias(mango, "mango", "english")

    tomato = kg.add_entity("টমেটো", "Tomato", "crop")
    kg.add_alias(tomato, "টমেটো", "standard")
    kg.add_alias(tomato, "tomato", "english")

    maize = kg.add_entity("ভুট্টা", "Maize", "crop")
    kg.add_alias(maize, "ভুট্টা", "standard")
    kg.add_alias(maize, "corn", "english")
    kg.add_alias(maize, "maize", "english")

    # =====================================================================
    # DISEASES
    # =====================================================================
    blast = kg.add_entity("ব্লাস্ট রোগ", "Rice Blast", "disease")
    kg.add_alias(blast, "ব্লাস্ট", "standard")
    kg.add_alias(blast, "ব্লাস্ট রোগ", "standard")
    kg.add_alias(blast, "blast", "english")
    kg.add_alias(blast, "rice blast", "english")
    kg.add_alias(blast, "পাতা পোড়া", "colloquial")  # leaf burning (colloquial)

    blight = kg.add_entity("ব্লাইট", "Bacterial Blight", "disease")
    kg.add_alias(blight, "ব্লাইট", "standard")
    kg.add_alias(blight, "bacterial blight", "english")
    kg.add_alias(blight, "blight", "english")
    kg.add_alias(blight, "পাতার দাগ রোগ", "colloquial")  # leaf spot disease

    sheath_blight = kg.add_entity("শীথ ব্লাইট", "Sheath Blight", "disease")
    kg.add_alias(sheath_blight, "শীথ ব্লাইট", "standard")
    kg.add_alias(sheath_blight, "sheath blight", "english")

    tungro = kg.add_entity("টুংরো", "Tungro", "disease")
    kg.add_alias(tungro, "টুংরো", "standard")
    kg.add_alias(tungro, "tungro", "english")
    kg.add_alias(tungro, "হলুদ রোগ", "colloquial")  # yellowing disease

    late_blight = kg.add_entity("নাবী ধসা", "Late Blight", "disease")
    kg.add_alias(late_blight, "নাবী ধসা", "standard")
    kg.add_alias(late_blight, "late blight", "english")
    kg.add_alias(late_blight, "আলুর রোগ", "colloquial")  # potato disease

    # =====================================================================
    # PESTS
    # =====================================================================
    stem_borer = kg.add_entity("মাজরা পোকা", "Stem Borer", "pest")
    kg.add_alias(stem_borer, "মাজরা পোকা", "standard")
    kg.add_alias(stem_borer, "মাজরা", "colloquial")
    kg.add_alias(stem_borer, "stem borer", "english")

    bph = kg.add_entity("বাদামি গাছফড়িং", "Brown Planthopper", "pest")
    kg.add_alias(bph, "বাদামি গাছফড়িং", "standard")
    kg.add_alias(bph, "গাছফড়িং", "colloquial")
    kg.add_alias(bph, "brown planthopper", "english")
    kg.add_alias(bph, "BPH", "english")

    leaf_roller = kg.add_entity("পাতা মোড়ানো পোকা", "Leaf Roller", "pest")
    kg.add_alias(leaf_roller, "পাতা মোড়ানো পোকা", "standard")
    kg.add_alias(leaf_roller, "leaf roller", "english")

    aphid = kg.add_entity("জাব পোকা", "Aphid", "pest")
    kg.add_alias(aphid, "জাব পোকা", "standard")
    kg.add_alias(aphid, "aphid", "english")

    # =====================================================================
    # SYMPTOMS
    # =====================================================================
    yellow_leaf = kg.add_entity("পাতা হলুদ হওয়া", "Leaf Yellowing", "symptom")
    kg.add_alias(yellow_leaf, "পাতা হলুদ", "colloquial")
    kg.add_alias(yellow_leaf, "হলুদ পাতা", "colloquial")
    kg.add_alias(yellow_leaf, "yellowing", "english")
    kg.add_alias(yellow_leaf, "yellow leaves", "english")

    leaf_spot = kg.add_entity("পাতায় দাগ", "Leaf Spots", "symptom")
    kg.add_alias(leaf_spot, "পাতায় দাগ", "standard")
    kg.add_alias(leaf_spot, "দাগ", "colloquial")
    kg.add_alias(leaf_spot, "leaf spot", "english")
    kg.add_alias(leaf_spot, "brown spots", "english")

    wilting = kg.add_entity("ঢলে পড়া", "Wilting", "symptom")
    kg.add_alias(wilting, "ঢলে পড়া", "standard")
    kg.add_alias(wilting, "গাছ শুকিয়ে যাওয়া", "colloquial")
    kg.add_alias(wilting, "wilting", "english")
    kg.add_alias(wilting, "wilt", "english")

    dead_heart = kg.add_entity("ডেড হার্ট", "Dead Heart", "symptom")
    kg.add_alias(dead_heart, "ডেড হার্ট", "standard")
    kg.add_alias(dead_heart, "মরা শীষ", "colloquial")
    kg.add_alias(dead_heart, "dead heart", "english")

    # =====================================================================
    # TREATMENTS / CHEMICALS
    # =====================================================================
    tricyclazole = kg.add_entity("ট্রাইসাইক্লাজল", "Tricyclazole", "chemical")
    kg.add_alias(tricyclazole, "ট্রাইসাইক্লাজল", "standard")
    kg.add_alias(tricyclazole, "tricyclazole", "english")

    carbendazim = kg.add_entity("কার্বেন্ডাজিম", "Carbendazim", "chemical")
    kg.add_alias(carbendazim, "কার্বেন্ডাজিম", "standard")
    kg.add_alias(carbendazim, "carbendazim", "english")

    imidacloprid = kg.add_entity("ইমিডাক্লোপ্রিড", "Imidacloprid", "chemical")
    kg.add_alias(imidacloprid, "ইমিডাক্লোপ্রিড", "standard")
    kg.add_alias(imidacloprid, "imidacloprid", "english")

    neem_oil = kg.add_entity("নিম তেল", "Neem Oil", "treatment")
    kg.add_alias(neem_oil, "নিম তেল", "standard")
    kg.add_alias(neem_oil, "নিমের তেল", "colloquial")
    kg.add_alias(neem_oil, "neem oil", "english")
    kg.add_alias(neem_oil, "neem", "english")

    # =====================================================================
    # FERTILIZERS
    # =====================================================================
    urea = kg.add_entity("ইউরিয়া", "Urea", "fertilizer")
    kg.add_alias(urea, "ইউরিয়া", "standard")
    kg.add_alias(urea, "urea", "english")

    tsp = kg.add_entity("টিএসপি", "TSP (Triple Super Phosphate)", "fertilizer")
    kg.add_alias(tsp, "টিএসপি", "standard")
    kg.add_alias(tsp, "TSP", "english")
    kg.add_alias(tsp, "triple super phosphate", "english")

    mop = kg.add_entity("এমওপি", "MOP (Muriate of Potash)", "fertilizer")
    kg.add_alias(mop, "এমওপি", "standard")
    kg.add_alias(mop, "MOP", "english")
    kg.add_alias(mop, "muriate of potash", "english")
    kg.add_alias(mop, "পটাশ", "colloquial")

    compost = kg.add_entity("জৈব সার", "Organic Compost", "fertilizer")
    kg.add_alias(compost, "জৈব সার", "standard")
    kg.add_alias(compost, "গোবর সার", "colloquial")  # cow dung fertilizer
    kg.add_alias(compost, "compost", "english")
    kg.add_alias(compost, "organic fertilizer", "english")

    # =====================================================================
    # RELATIONS
    # =====================================================================

    # Symptoms → Diseases
    kg.add_relation(leaf_spot, "symptom_of", blast, "IRRI rice manual")
    kg.add_relation(yellow_leaf, "symptom_of", tungro, "IRRI rice manual")
    kg.add_relation(yellow_leaf, "symptom_of", blight, "FAO pest manual")
    kg.add_relation(dead_heart, "symptom_of", stem_borer, "IRRI rice manual")
    kg.add_relation(wilting, "symptom_of", late_blight, "FAO manual")

    # Diseases → Crops
    kg.add_relation(blast, "affects", rice, "IRRI rice manual")
    kg.add_relation(blight, "affects", rice, "IRRI rice manual")
    kg.add_relation(sheath_blight, "affects", rice, "IRRI rice manual")
    kg.add_relation(tungro, "affects", rice, "IRRI rice manual")
    kg.add_relation(late_blight, "affects", potato, "FAO manual")

    # Pests → Crops
    kg.add_relation(stem_borer, "attacks", rice, "IRRI rice manual")
    kg.add_relation(bph, "attacks", rice, "IRRI rice manual")
    kg.add_relation(leaf_roller, "attacks", rice, "IRRI rice manual")

    # Treatments → Diseases/Pests
    kg.add_relation(tricyclazole, "treatment_for", blast, "IRRI rice manual")
    kg.add_relation(carbendazim, "treatment_for", sheath_blight, "IRRI rice manual")
    kg.add_relation(imidacloprid, "treatment_for", bph, "IRRI rice manual")
    kg.add_relation(neem_oil, "treatment_for", aphid, "FAO pest manual")

    # Fertilizers → Crops
    kg.add_relation(urea, "applied_to", rice, "general")
    kg.add_relation(tsp, "applied_to", rice, "general")
    kg.add_relation(mop, "applied_to", rice, "general")

    stats = kg.get_stats()
    logger.info(
        "KG seeded: %d entities, %d aliases, %d relations",
        stats["entities"],
        stats["aliases"],
        stats["relations"],
    )
