// Class table definition for self-describing modules.
// Include after `table_abi.h`.

static const char kGpClassTableJson[] =
"{\n"
"  \"table\": {\n"
"    \"name\": \"class_def\",\n"
"    \"columns\": [\n"
"      {\"kind\": \"text\", \"width\": 120, \"align\": 0},\n"
"      {\"kind\": \"text\", \"width\": 120, \"align\": 0},\n"
"      {\"kind\": \"text\", \"width\": 160, \"align\": 0}\n"
"    ],\n"
"    \"rows\": [\n"
"      {\"kind\": \"header\", \"label\": \"Fields\", \"depth\": 0, \"expanded\": true,\n"
"       \"cells\": [\"Type\", \"Default\", \"Notes\"]},\n"
"      {\"kind\": \"note\", \"label\": \"gain\", \"depth\": 1,\n"
"       \"cells\": [\"float\", \"1.0\", \"input\"]},\n"
"      {\"kind\": \"note\", \"label\": \"enabled\", \"depth\": 1,\n"
"       \"cells\": [\"bool\", \"true\", \"toggle\"]},\n"
"      {\"kind\": \"button\", \"label\": \"Add Field\", \"depth\": 1,\n"
"       \"cells\": [\"\", \"\", \"\"]},\n"
"      {\"kind\": \"header\", \"label\": \"Methods\", \"depth\": 0, \"expanded\": true,\n"
"       \"cells\": [\"Args\", \"Returns\", \"Notes\"]},\n"
"      {\"kind\": \"note\", \"label\": \"step\", \"depth\": 1,\n"
"       \"cells\": [\"dt\", \"void\", \"tick\"]},\n"
"      {\"kind\": \"button\", \"label\": \"Add Method\", \"depth\": 1,\n"
"       \"cells\": [\"\", \"\", \"\"]}\n"
"    ],\n"
"    \"actions\": [\n"
"      {\"id\": \"edit_field\", \"row\": 1, \"col\": -1, \"part\": \"cell\"},\n"
"      {\"id\": \"edit_field\", \"row\": 2, \"col\": -1, \"part\": \"cell\"},\n"
"      {\"id\": \"add_field\", \"row\": 3, \"col\": -1, \"part\": \"cell\"},\n"
"      {\"id\": \"edit_method\", \"row\": 5, \"col\": -1, \"part\": \"cell\"},\n"
"      {\"id\": \"add_method\", \"row\": 6, \"col\": -1, \"part\": \"cell\"}\n"
"    ]\n"
"  }\n"
"}\n";

enum GP_ClassTableActionId {
    GP_CLASS_ACT_EDIT_FIELD = 1,
    GP_CLASS_ACT_ADD_FIELD = 2,
    GP_CLASS_ACT_EDIT_METHOD = 3,
    GP_CLASS_ACT_ADD_METHOD = 4
};

#define GP_TABLE_CELL_TEXT_INIT(str) { GP_TABLE_CELL_TEXT, str, 0u, 0.0f, 0.0f, 0.0f, 0, 0 }
#define GP_TABLE_CELL_EMPTY { GP_TABLE_CELL_TEXT, "", 0u, 0.0f, 0.0f, 0.0f, 0, 0 }

static const GP_TableColumn kGpClassTableColumns[] = {
    { GP_TABLE_CELL_TEXT, 120, 0 },
    { GP_TABLE_CELL_TEXT, 120, 0 },
    { GP_TABLE_CELL_TEXT, 160, 0 }
};

static const GP_TableRow kGpClassTableRows[] = {
    { GP_TABLE_ROW_HEADER, 0, 1, 0, "Fields",
      { GP_TABLE_CELL_TEXT_INIT("Type"),
        GP_TABLE_CELL_TEXT_INIT("Default"),
        GP_TABLE_CELL_TEXT_INIT("Notes"),
        GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY },
      3, 0 },
    { GP_TABLE_ROW_NOTE, 1, 0, 0, "gain",
      { GP_TABLE_CELL_TEXT_INIT("float"),
        GP_TABLE_CELL_TEXT_INIT("1.0"),
        GP_TABLE_CELL_TEXT_INIT("input"),
        GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY },
      3, 0 },
    { GP_TABLE_ROW_NOTE, 1, 0, 0, "enabled",
      { GP_TABLE_CELL_TEXT_INIT("bool"),
        GP_TABLE_CELL_TEXT_INIT("true"),
        GP_TABLE_CELL_TEXT_INIT("toggle"),
        GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY },
      3, 0 },
    { GP_TABLE_ROW_BUTTON, 1, 0, 0, "Add Field",
      { GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY,
        GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY },
      3, 0 },
    { GP_TABLE_ROW_HEADER, 0, 1, 0, "Methods",
      { GP_TABLE_CELL_TEXT_INIT("Args"),
        GP_TABLE_CELL_TEXT_INIT("Returns"),
        GP_TABLE_CELL_TEXT_INIT("Notes"),
        GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY },
      3, 0 },
    { GP_TABLE_ROW_NOTE, 1, 0, 0, "step",
      { GP_TABLE_CELL_TEXT_INIT("dt"),
        GP_TABLE_CELL_TEXT_INIT("void"),
        GP_TABLE_CELL_TEXT_INIT("tick"),
        GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY },
      3, 0 },
    { GP_TABLE_ROW_BUTTON, 1, 0, 0, "Add Method",
      { GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY,
        GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY, GP_TABLE_CELL_EMPTY },
      3, 0 }
};

static const GP_TableAction kGpClassTableActions[] = {
    { 1, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, GP_TABLE_ACTION_ANY, GP_CLASS_ACT_EDIT_FIELD },
    { 2, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, GP_TABLE_ACTION_ANY, GP_CLASS_ACT_EDIT_FIELD },
    { 3, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, GP_TABLE_ACTION_ANY, GP_CLASS_ACT_ADD_FIELD },
    { 5, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, GP_TABLE_ACTION_ANY, GP_CLASS_ACT_EDIT_METHOD },
    { 6, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_CELL, GP_TABLE_ACTION_ANY, GP_CLASS_ACT_ADD_METHOD }
};

static const GP_TableObjectDef kGpClassTableDef = {
    kGpClassTableColumns,
    (int32_t)(sizeof(kGpClassTableColumns) / sizeof(kGpClassTableColumns[0])),
    kGpClassTableRows,
    (int32_t)(sizeof(kGpClassTableRows) / sizeof(kGpClassTableRows[0])),
    kGpClassTableActions,
    (int32_t)(sizeof(kGpClassTableActions) / sizeof(kGpClassTableActions[0]))
};

#undef GP_TABLE_CELL_TEXT_INIT
#undef GP_TABLE_CELL_EMPTY
