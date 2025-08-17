test=[
    {
      "id": "RFC7952-ANN-DEF-CARDINALITY",
      "source": "RFC7952 \u00a73 (Table 2) \ue200filecite\ue202turn1file15\ue201",
      "level": "MUST",
      "applies_to": "md:annotation",
      "rule": "Each md:annotation definition MUST contain exactly one 'type' substatement; optional substatements are: description (0..1), if-feature (0..n), reference (0..1), status (0..1), units (0..1).",
      "rationale": "Ensures annotation definitions are well formed and types are enforced.",
      "autofix_hint": "If 'type' is missing, insert a suitable built-in or derived type consistent with the intended semantics; remove any disallowed substatements."
    },
    {
      "id": "RFC7952-ANN-SCALAR-ONLY",
      "source": "RFC7952 \u00a72 (end of p.4) \ue200filecite\ue202turn1file16\ue201",
      "level": "MUST NOT",
      "applies_to": "Annotation usage",
      "rule": "Annotations MUST be scalar values and MUST NOT be attached to an entire list or leaf-list instance\u2014only to individual list or leaf-list entries.",
      "rationale": "XML/JSON encoding compatibility constraints.",
      "autofix_hint": "If an annotation targets a list/leaf-list as a whole, move it to each affected entry (or to an appropriate container node) with scalar values."
    },
    {
      "id": "RFC7952-ANN-NO-SEMANTIC-CHANGE",
      "source": "RFC7952 \u00a73 (after Table 2) \ue200filecite\ue202turn1file11\ue201",
      "level": "MUST NOT",
      "applies_to": "Annotation definitions",
      "rule": "An annotation MUST NOT change YANG data tree semantics (e.g., cannot override uniqueness of leaf-list entries).",
      "rationale": "Annotations add metadata and must not alter core semantics.",
      "autofix_hint": "Remove or rework annotations that attempt to enforce semantics doable by core YANG statements (use 'must', 'unique', etc. instead)."
    },
    {
      "id": "RFC7952-ANN-MODULE-PURITY",
      "source": "RFC7952 \u00a73 (after Table 2) \ue200filecite\ue202turn1file11\ue201",
      "level": "SHOULD NOT",
      "applies_to": "Modules defining md:annotation",
      "rule": "A module containing md:annotation statements SHOULD NOT also define data nodes or groupings; derived types, identities, and features SHOULD NOT be defined unless used by the annotations.",
      "rationale": "Keep annotation modules focused and reusable.",
      "autofix_hint": "Move regular data definitions to a separate module unless directly required by the annotations."
    },
    {
      "id": "RFC7952-ANN-ADVERTISE-BEFORE-USE",
      "source": "RFC7952 \u00a74 \ue200filecite\ue202turn1file11\ue201",
      "level": "MUST NOT",
      "applies_to": "Clients/instances",
      "rule": "Clients MUST NOT add an annotation unless the server advertises support by exporting the module that defines it.",
      "rationale": "Prevents interop failures from unrecognized annotations.",
      "autofix_hint": "Before emitting annotations, ensure server capability discovery includes the annotation\u2019s module; otherwise omit."
    },
    {
      "id": "RFC7952-JSON-ANN-NAMING",
      "source": "RFC7952 \u00a75.2.1 \ue200filecite\ue202turn1file5\ue201",
      "level": "MUST",
      "applies_to": "JSON instance encoding",
      "rule": "Each annotation member name MUST be 'module-name \":\" annotation-name' (explicit module name is always required for annotations).",
      "rationale": "Disambiguates annotation namespaces in JSON.",
      "autofix_hint": "Prefix bare annotation names with their defining module and a colon."
    },
    {
      "id": "RFC7952-JSON-ANN-VALUE-ENCODING",
      "source": "RFC7952 \u00a75.2.1 \ue200filecite\ue202turn1file5\ue201",
      "level": "SHALL",
      "applies_to": "JSON instance encoding",
      "rule": "The value of each annotation SHALL be encoded exactly as a leaf with the same YANG type (per RFC7951).",
      "rationale": "Ensures consistent scalar encoding across data and annotations.",
      "autofix_hint": "Apply RFC7951 rules for the annotation\u2019s YANG type."
    },
    {
      "id": "RFC7952-JSON-ANN-CONTAINER-LIST-ANYDATA",
      "source": "RFC7952 \u00a75.2.2 \ue200filecite\ue202turn1file5\ue201",
      "level": "MUST",
      "applies_to": "JSON instance encoding",
      "rule": "For container, list entry, or anydata, attach a metadata object as the '@' member within that object.",
      "rationale": "Standard placement of annotation metadata object.",
      "autofix_hint": "Insert an '@' object sibling to other members and place annotation pairs within it."
    },
    {
      "id": "RFC7952-JSON-ANN-LEAF-ANYXML",
      "source": "RFC7952 \u00a75.2.3 \ue200filecite\ue202turn1file17\ue201",
      "level": "MUST",
      "applies_to": "JSON instance encoding",
      "rule": "For leaf or anyxml, add a sibling member named '@' + <member-name> containing the metadata object; include the module prefix in the '@' name iff present in the annotated member name.",
      "rationale": "Correct sibling placement and naming for scalar/anyxml.",
      "autofix_hint": "Create '@<name>' (or '@mod:<name>') member adjacent to the value with the metadata object."
    },
    {
      "id": "RFC7952-JSON-ANN-LEAFLIST",
      "source": "RFC7952 \u00a75.2.4 \ue200filecite\ue202turn1file18\ue201",
      "level": "MUST",
      "applies_to": "JSON instance encoding",
      "rule": "For a leaf-list, add sibling '@<leaf-list-name>' whose value is an array mapping 1:1 by index to entries; each element is a metadata object or null; trailing nulls MAY be omitted.",
      "rationale": "Preserves positional mapping of annotations to entries.",
      "autofix_hint": "Construct an annotation array matching the leaf-list length; fill unused indices with null and drop trailing nulls."
    },
    {
      "id": "RFC7952-ANN-SECURITY-PARITY",
      "source": "RFC7952 \u00a79 \ue200filecite\ue202turn1file14\ue201",
      "level": "SHOULD",
      "applies_to": "Access control",
      "rule": "Annotations SHOULD be protected with the same or stricter access control as their target data node.",
      "rationale": "Annotations may reveal sensitive metadata.",
      "autofix_hint": "Align NACM rules for annotation paths with target node ACLs or stronger."
    },
    {
      "id": "RFC8407-PREFIX-USAGE-EXTERNAL",
      "source": "RFC8407 \u00a74.2 (prefix rules) \ue200filecite\ue202turn1file2\ue201",
      "level": "MUST",
      "applies_to": "Module text",
      "rule": "A prefix MUST be used for any external statement (defined via 'extension') and for all identifiers imported from other modules and included from submodules.",
      "rationale": "Avoids ambiguity across module boundaries.",
      "autofix_hint": "Qualify external/foreign identifiers with the correct module prefix."
    },
    {
      "id": "RFC8407-PREFIX-USAGE-LOCAL",
      "source": "RFC8407 \u00a74.2 \ue200filecite\ue202turn1file2\ue201",
      "level": "MUST/SHOULD",
      "applies_to": "Module text",
      "rule": "The local module prefix MUST be used in all 'default' statements for identityref or instance-identifier; it SHOULD be used in all path expressions.",
      "rationale": "Prevents ambiguity in defaults and improves clarity in paths.",
      "autofix_hint": "Add local prefix to identityref/instance-identifier defaults; qualify unprefixed names in XPath-like expressions."
    },
    {
      "id": "RFC8407-ID-LENGTH",
      "source": "RFC8407 \u00a74.3 \ue200filecite\ue202turn1file3\ue201",
      "level": "MUST",
      "applies_to": "All identifiers",
      "rule": "Identifiers in published modules MUST be 1..64 characters (per 'identifier-arg-str').",
      "rationale": "Ensures interoperability with tooling and specs.",
      "autofix_hint": "Shorten identifiers exceeding 64 chars; avoid empty names."
    },
    {
      "id": "RFC8407-ID-NAMING-CONVENTIONS",
      "source": "RFC8407 \u00a74.3.1 \ue200filecite\ue202turn1file3\ue201",
      "level": "SHOULD",
      "applies_to": "All identifiers",
      "rule": "Prefer lowercase letters, digits, and dashes; avoid repeating parent identifiers; use full words/acronyms; avoid embedding modeling semantics in names.",
      "rationale": "Improves readability and consistency.",
      "autofix_hint": "Rename to kebab-case; remove redundant parent terms; avoid names like 'config' or 'state' for semantics."
    },
    {
      "id": "RFC8407-DEFAULTS-OMIT-COMMON",
      "source": "RFC8407 \u00a74.4 (Statement Defaults table) \ue200filecite\ue202turn1file3\ue201",
      "level": "SHOULD NOT",
      "applies_to": "Module style",
      "rule": "Avoid specifying substatements when using their common default values: config=true, mandatory=false, max-elements=unbounded, min-elements=0, ordered-by=system, status=current, yin-element=false.",
      "rationale": "Reduces noise; improves readability.",
      "autofix_hint": "Remove redundant defaulted substatements unless doing so would reduce clarity."
    },
    {
      "id": "RFC8407-FEATURE-DESIGN",
      "source": "RFC8407 \u00a74.17 (features) \ue200filecite\ue202turn1file9\ue201",
      "level": "MUST/SHOULD",
      "applies_to": "feature / if-feature",
      "rule": "Within each 'feature', the 'description' MUST specify interactions with other features; avoid overly fine-grained features; if one feature requires another, add 'if-feature' in the dependent feature.",
      "rationale": "Promotes coherent, discoverable feature sets.",
      "autofix_hint": "Consolidate leaf-level features; add explicit dependency via 'if-feature'."
    },
    {
      "id": "RFC8407-WHEN-VS-MUST",
      "source": "RFC8407 \u00a74.18.2 \ue200filecite\ue202turn1file9\ue201",
      "level": "SHOULD",
      "applies_to": "Constraint selection",
      "rule": "Use 'when' with 'augment' or 'uses' for conditional composition based on static properties (e.g., keys); remember that when=false deletes nodes silently and is not an error.",
      "rationale": "Correct application of cross-object constraints.",
      "autofix_hint": "Where conditional presence is intended, prefer 'when' on the augmented node; reserve 'must' for validating present data."
    },
    {
      "id": "RFC8407-SEC-CONSIDERATIONS-TEMPLATE",
      "source": "RFC8407 \u00a73.7\u2013\u00a73.7.1 \ue200filecite\ue202turn1file10\ue202turn1file13\ue201",
      "level": "MUST",
      "applies_to": "Specifications publishing modules",
      "rule": "A specification defining modules MUST include a Security Considerations section patterned after the current IETF YANG security template; explicitly list sensitive writable/readable nodes and potentially harmful RPCs by name.",
      "rationale": "Mandatory IETF process guidance for modules in specs.",
      "autofix_hint": "Generate security text from the model inventory of 'config true' nodes, sensitive read nodes, and RPCs; include NACM/TLS/SSH references per template."
    }
  ]

import json
transformed_data = []
for obj in test:
    new_obj = {k: v for k, v in obj.items() if k not in ["id", "source"]}
    if "level" in new_obj:
        new_obj["name"] = new_obj.pop("level")
    transformed_data.append(new_obj)

print(json.dumps(transformed_data, indent=2))