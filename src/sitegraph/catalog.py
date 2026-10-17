"""Every relationship type a device graph may contain, with the query that uses it (spec §5 governance).

A type is added here by the change that first creates it, together with the query that needs it; a test fails when the
graph holds a type that is not listed, or when a listed type is used nowhere in the code."""

REL_CATALOG = {
    "HAS_STATE": "repair re-crawl of a page's states; explanations show the page",
    "COMPOSED_OF": "executor reads current component states for ACTION.pre_cstates",
    "HAS_CSTATE": "impact analysis; healing scope; transitions",
    "CONTAINS": "impact analysis; frontier counts; storyboard transition lookup",
    "ACTION": "routes (algo.SPpaths), next actions, BFS frontier, PageRank/betweenness/CDLP",
    "TRANSITION": "pre_cstates satisfaction; micro demo shots",
    "BROKEN_ACTION": "quarantined actions (excluded from algorithms); repair",
    "FULFILLED_BY": "intent resolution",
    "STEP_OF": "workflow membership; demo score; impact",
    "FROM": "step anchors (WStep, Step)",
    "TO": "where a step actually landed (drift evidence)",
    "ON": "workflow step element (impact analysis)",
    "IN_RUN": "run membership; retention",
    "NEXT": "step sequence mining",
    "EXECUTED": "workflow usage; retention protection",
    "DURING": "observation evidence per step",
    "OF": "observation target",
    "PRODUCED": "demo request -> storyboard",
    "BASED_ON": "storyboard -> workflow (impact, re-render)",
    "IN_STORYBOARD": "shot membership",
    "SHOWS": "shot -> state (impact, re-render)",
    "FROM_STEP": "shot provenance; retention protection",
    "RENDERED_AS": "storyboard -> render files",
}
