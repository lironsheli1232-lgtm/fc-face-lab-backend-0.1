import express from "express";
import cors from "cors";
import { GoogleGenAI } from "@google/genai";

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: "40mb" }));

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY
});

const ANALYSIS_MODEL = "gemini-3.8-flash";

async function withRetry(fn, attempts = 4) {
  let lastError;

  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      const status =
        error?.status ||
        error?.code ||
        error?.response?.status;

      const message =
        String(error?.message || "").toLowerCase();

      const temporary =
        status === 503 ||
        status === 429 ||
        message.includes("high demand") ||
        message.includes("unavailable") ||
        message.includes("overloaded") ||
        message.includes("resource_exhausted");

      if (!temporary || attempt === attempts) {
        throw error;
      }

      const wait =
        attempt === 1 ? 2500 :
        attempt === 2 ? 5000 :
        9000;

      await new Promise(resolve => setTimeout(resolve, wait));
    }
  }

  throw lastError;
}

/*
========================================================
FC FACE LAB — MULTI PHOTO FACE ANALYSIS ENGINE
========================================================

The system accepts multiple photographs of the same person.

Recommended:
1. Front
2. 45 degree
3. Profile
4. Additional clear photo

The model must combine all photographs into ONE
unified facial geometry estimate.

It must NOT independently describe each photo.

========================================================
*/

const CREATOR_KNOWLEDGE = {

  philosophy: `
The workflow used by EA FC face-creation creators is:

1. Select the closest base face.
2. Establish the overall head/skull shape.
3. Adjust forehead and temples.
4. Adjust brow and eye region.
5. Adjust nose.
6. Adjust cheeks.
7. Adjust jaw.
8. Adjust mouth.
9. Adjust chin.
10. Adjust ears.
11. Finish skin, hair, eyebrows and facial hair.
12. Re-check the complete face.

The important concept is progressive refinement:
large structural changes first, small facial details afterward.
`,

  officialCranium: {
    confirmedControls: [
      "upper forehead width",
      "eye socket depth",
      "jaw position",
      "skull width",
      "temple size"
    ],

    confirmedFeatures: [
      "Basic Sculpt",
      "Advanced Sculpt",
      "face presets",
      "head model sculpting",
      "complexion and texture customization"
    ]
  },

  communityMethod: `
Community face-creation tutorials commonly use a base face first,
followed by progressive sculpting of individual facial regions.

Do NOT treat any individual creator's settings as universal.

Instead learn the relationship:

reference face
→ facial geometry
→ base face choice
→ sculpt adjustments
→ final appearance.

Examples from tutorials are calibration references, not rules.
`,

  geometryCategories: [
    "head",
    "forehead",
    "temples",
    "brow",
    "eyes",
    "nose",
    "cheeks",
    "jaw",
    "chin",
    "mouth",
    "ears"
  ]
};

const FC27_CONTROLS = {

  head: [
    "skullWidth",
    "skullHeight",
    "skullDepth",
    "headWeight",
    "headDefinition"
  ],

  forehead: [
    "upperForeheadWidth",
    "foreheadHeight",
    "foreheadProjection"
  ],

  temples: [
    "templeSize",
    "templeWidth",
    "templeDepth"
  ],

  brow: [
    "browWidth",
    "browHeight",
    "browDepth",
    "browProjection",
    "browAngle",
    "browSpacing"
  ],

  eyes: [
    "eyeSize",
    "eyeSpacing",
    "eyeHeight",
    "eyeDepth",
    "eyeSocketDepth",
    "eyeAngle",
    "upperEyelid",
    "lowerEyelid"
  ],

  nose: [
    "noseWidth",
    "noseHeight",
    "noseLength",
    "noseProjection",
    "bridgeWidth",
    "bridgeHeight",
    "bridgeProjection",
    "tipWidth",
    "tipHeight",
    "tipProjection",
    "nostrilWidth",
    "nostrilHeight"
  ],

  cheeks: [
    "cheekWidth",
    "cheekHeight",
    "cheekProjection",
    "cheekFullness",
    "cheekDefinition"
  ],

  jaw: [
    "jawWidth",
    "jawHeight",
    "jawDepth",
    "jawProjection",
    "jawDefinition",
    "jawAngle",
    "jawPosition"
  ],

  chin: [
    "chinWidth",
    "chinHeight",
    "chinProjection",
    "chinDepth",
    "chinFullness",
    "chinDefinition"
  ],

  mouth: [
    "mouthWidth",
    "mouthHeight",
    "mouthPosition",
    "mouthProjection",
    "upperLipThickness",
    "lowerLipThickness",
    "lipFullness",
    "lipCornerAngle"
  ],

  ears: [
    "earSize",
    "earHeight",
    "earProjection",
    "earAngle",
    "earLobeSize"
  ]
};

function clamp(value) {
  const n = Number(value);

  if (!Number.isFinite(n)) {
    return null;
  }

  return Math.max(0, Math.min(100, Math.round(n)));
}

function normalizeControl(control) {

  if (!control || typeof control !== "object") {
    return null;
  }

  return {
    value: clamp(control.value),

    direction:
      typeof control.direction === "string"
        ? control.direction
        : "",

    description:
      typeof control.description === "string"
        ? control.description
        : "",

    confidence:
      clamp(control.confidence)
  };
}

function normalizeAnalysis(raw) {

  const output = {

    version: "FC-FACE-LAB-MULTI-PHOTO-3.0",

    input: {
      photoCount: 0,
      analysisMethod:
        "Multi-photo unified facial geometry analysis"
    },

    baseFace: {
      preset: null,
      description: "",
      confidence: null,
      reason: ""
    },

    sculpt: {},

    appearance: {
      skinTone: "",
      complexion: "",
      eyeColor: "",
      eyebrowStyle: "",
      hairstyle: "",
      hairColor: "",
      facialHair: ""
    },

    symmetry: {
      overall: "",
      importantDifferences: []
    },

    instructions: [],

    confidence: {
      overall: null
    },

    warnings: []
  };

  if (raw?.input) {
    output.input.photoCount =
      Number(raw.input.photoCount) || 0;
  }

  if (raw?.baseFace) {

    output.baseFace = {
      preset:
        raw.baseFace.preset ?? null,

      description:
        typeof raw.baseFace.description === "string"
          ? raw.baseFace.description
          : "",

      confidence:
        clamp(raw.baseFace.confidence),

      reason:
        typeof raw.baseFace.reason === "string"
          ? raw.baseFace.reason
          : ""
    };
  }

  for (const category of Object.keys(FC27_CONTROLS)) {

    const source =
      raw?.sculpt?.[category];

    if (!source) {
      continue;
    }

    output.sculpt[category] = {
      name:
        typeof source.name === "string"
          ? source.name
          : category,

      confidence:
        clamp(source.confidence),

      controls: {}
    };

    for (const control of FC27_CONTROLS[category]) {

      if (!source.controls?.[control]) {
        continue;
      }

      const normalized =
        normalizeControl(
          source.controls[control]
        );

      if (normalized) {
        output.sculpt[category]
          .controls[control] = normalized;
      }
    }
  }

  if (raw?.appearance) {
    output.appearance = {
      skinTone: raw.appearance.skinTone || "",
      complexion: raw.appearance.complexion || "",
      eyeColor: raw.appearance.eyeColor || "",
      eyebrowStyle: raw.appearance.eyebrowStyle || "",
      hairstyle: raw.appearance.hairstyle || "",
      hairColor: raw.appearance.hairColor || "",
      facialHair: raw.appearance.facialHair || ""
    };
  }

  if (raw?.symmetry) {
    output.symmetry = {
      overall:
        raw.symmetry.overall || "",

      importantDifferences:
        Array.isArray(
          raw.symmetry.importantDifferences
        )
          ? raw.symmetry.importantDifferences.slice(0, 10)
          : []
    };
  }

  if (Array.isArray(raw?.instructions)) {
    output.instructions =
      raw.instructions.slice(0, 80);
  }

  if (raw?.confidence) {
    output.confidence = {
      overall:
        clamp(raw.confidence.overall)
    };
  }

  if (Array.isArray(raw?.warnings)) {
    output.warnings =
      raw.warnings.slice(0, 20);
  }

  return output;
}

app.get("/", (req, res) => {

  res.json({
    ok: true,
    service: "FC FACE LAB",
    version: "3.0",
    status: "running"
  });
});

app.post("/analyze", async (req, res) => {

  try {

    if (!process.env.GEMINI_API_KEY) {

      return res.status(500).json({
        success: false,
        error: "GEMINI_API_KEY is missing"
      });
    }

    const {
      images,
      image,
      mimeType
    } = req.body;

    /*
    Backward compatibility:
    the old frontend can still send one image.
    */

    let photoList = [];

    if (
      Array.isArray(images) &&
      images.length > 0
    ) {

      photoList = images
        .filter(item =>
          item &&
          item.data &&
          item.mimeType
        )
        .slice(0, 8);

    } else if (image && mimeType) {

      photoList = [
        {
          data: image,
          mimeType
        }
      ];
    }

    if (photoList.length === 0) {

      return res.status(400).json({
        success: false,
        error: "At least one face image is required"
      });
    }

    const imageParts =
      photoList.map((photo, index) => ({
        inlineData: {
          mimeType: photo.mimeType,
          data: photo.data
        }
      }));

    const analysisPrompt = `

You are the core facial reconstruction engine for
FC FACE LAB.

Your task is to convert multiple photographs of ONE
person into practical EA SPORTS FC 27 face-creation
instructions.

This is NOT a generic face-description task.

The user wants to recreate themselves inside
EA SPORTS FC 27.

====================================================
MULTI-PHOTO ANALYSIS
====================================================

You are receiving ${photoList.length} photograph(s).

Treat every photograph as evidence of the SAME person.

DO NOT analyze each image independently and average
the descriptions.

Instead construct ONE unified mental 3D representation
of the person's face.

Use:

FRONT VIEW:
- facial width
- symmetry
- eye spacing
- nose width
- mouth width
- jaw width
- chin width

45 DEGREE VIEW:
- nose projection
- cheek projection
- jaw projection
- chin projection
- forehead projection
- temple structure

PROFILE:
- nose length
- nose projection
- forehead projection
- lips projection
- chin projection
- jaw position
- overall head depth

ADDITIONAL PHOTOS:
Use them to resolve uncertainty.

If two photographs disagree because of perspective,
lens distortion, expression, lighting or pose,
determine which measurement is more likely to represent
the underlying facial structure.

Do not simply average incompatible measurements.

====================================================
PHOTO QUALITY
====================================================

For each major facial region consider:

- visibility
- camera angle
- lighting
- expression
- lens distortion
- confidence

If a feature cannot be determined reliably,
lower confidence.

Never manufacture certainty.

====================================================
EA SPORTS FC / CRANIUM WORKFLOW
====================================================

The workflow should resemble professional community
Face Creation tutorials:

1. Base face.
2. Large head structure.
3. Forehead and temples.
4. Brow.
5. Eyes and eye sockets.
6. Nose.
7. Cheeks.
8. Jaw.
9. Mouth.
10. Chin.
11. Ears.
12. Skin/complexion.
13. Hair.
14. Facial hair.
15. Final refinement.

Large structural changes must be solved before
small details.

====================================================
IMPORTANT KNOWLEDGE RULE
====================================================

The following are known Cranium concepts:

- Basic Sculpt
- Advanced Sculpt
- Upper Forehead Width
- Eye Socket Depth
- Jaw Position
- Skull Width
- Temple Size

These are supported by EA documentation.

Community Face Creation tutorials are useful as
CALIBRATION REFERENCES for the workflow.

Do not pretend that a community tutorial proves that
every similarly named control exists in FC27.

Do not invent official EA terminology.

====================================================
NUMERICAL SYSTEM
====================================================

Whenever the exact FC27 numerical range is not publicly
documented, output a NORMALIZED 0-100 TARGET.

Interpretation:

0   = minimum
50  = neutral / center
100 = maximum

This is a practical target, NOT an official EA value.

Every number must represent the direction in which the
user should move the corresponding creator control.

Do not make everything approximately 50.

Use extreme values when the facial geometry genuinely
requires them.

====================================================
BASE FACE
====================================================

Choose the closest base face ONLY when there is enough
evidence.

If you cannot identify the exact preset confidently:

preset = null

Then describe the visual characteristics of the base
face the user should choose.

Do NOT hallucinate preset numbers.

====================================================
HEAD
====================================================

Analyze:

- skull width
- skull height
- skull depth
- head weight
- head definition
- upper forehead width
- forehead height
- forehead projection
- temple size
- temple width
- temple depth

====================================================
BROW
====================================================

Analyze:

- brow width
- brow height
- brow depth
- brow projection
- brow angle
- brow spacing

====================================================
EYES
====================================================

Analyze:

- eye size
- eye spacing
- eye height
- eye depth
- eye socket depth
- eye angle
- upper eyelid
- lower eyelid

====================================================
NOSE
====================================================

Analyze:

- overall width
- height
- length
- projection
- bridge width
- bridge height
- bridge projection
- tip width
- tip height
- tip projection
- nostril width
- nostril height

Use the 45-degree and profile photos heavily here.

====================================================
CHEEKS
====================================================

Analyze:

- width
- height
- projection
- fullness
- definition

====================================================
JAW
====================================================

Analyze:

- width
- height
- depth
- projection
- definition
- angle
- position

====================================================
CHIN
====================================================

Analyze:

- width
- height
- projection
- depth
- fullness
- definition

====================================================
MOUTH
====================================================

Analyze:

- width
- height
- position
- projection
- upper lip
- lower lip
- fullness
- corners

====================================================
EARS
====================================================

Analyze:

- size
- height
- projection
- angle
- lobe

====================================================
ASYMMETRY
====================================================

Do not automatically make the face symmetrical.

Look for:

- eyebrow differences
- eye height differences
- nostril differences
- jaw asymmetry
- cheek asymmetry
- chin offset
- mouth corner differences

Only report asymmetry when supported by the photos.

====================================================
APPEARANCE
====================================================

Analyze:

- skin tone
- complexion
- eyes
- eyebrows
- hair
- facial hair

Keep these separate from bone structure.

====================================================
INSTRUCTIONS
====================================================

Create practical instructions in the exact order:

BASE FACE

HEAD

FOREHEAD

TEMPLES

BROW

EYES

NOSE

CHEEKS

JAW

MOUTH

CHIN

EARS

SKIN

HAIR

FACIAL HAIR

FINAL CHECK

Every important instruction should contain:

- category
- control
- target value 0-100
- direction
- explanation
- confidence

Example:

{
  "category": "nose",
  "control": "noseWidth",
  "target": 64,
  "direction": "increase moderately",
  "instruction": "Move the nose width slider toward wider until approximately 64/100.",
  "confidence": 86
}

====================================================
OUTPUT
====================================================

Return ONLY valid JSON.

Use this structure:

{
  "input": {
    "photoCount": ${photoList.length}
  },

  "baseFace": {
    "preset": null,
    "description": "",
    "confidence": 0,
    "reason": ""
  },

  "sculpt": {

    "head": {
      "name": "Head / Skull",
      "confidence": 0,
      "controls": {}
    },

    "forehead": {
      "name": "Forehead",
      "confidence": 0,
      "controls": {}
    },

    "temples": {
      "name": "Temples",
      "confidence": 0,
      "controls": {}
    },

    "brow": {
      "name": "Brow",
      "confidence": 0,
      "controls": {}
    },

    "eyes": {
      "name": "Eyes / Eye Socket",
      "confidence": 0,
      "controls": {}
    },

    "nose": {
      "name": "Nose",
      "confidence": 0,
      "controls": {}
    },

    "cheeks": {
      "name": "Cheeks",
      "confidence": 0,
      "controls": {}
    },

    "jaw": {
      "name": "Jaw",
      "confidence": 0,
      "controls": {}
    },

    "chin": {
      "name": "Chin",
      "confidence": 0,
      "controls": {}
    },

    "mouth": {
      "name": "Mouth",
      "confidence": 0,
      "controls": {}
    },

    "ears": {
      "name": "Ears",
      "confidence": 0,
      "controls": {}
    }
  },

  "appearance": {
    "skinTone": "",
    "complexion": "",
    "eyeColor": "",
    "eyebrowStyle": "",
    "hairstyle": "",
    "hairColor": "",
    "facialHair": ""
  },

  "symmetry": {
    "overall": "",
    "importantDifferences": []
  },

  "instructions": [],

  "confidence": {
    "overall": 0
  },

  "warnings": []
}

Use the following knowledge:

${JSON.stringify(CREATOR_KNOWLEDGE, null, 2)}

Available practical control mapping:

${JSON.stringify(FC27_CONTROLS, null, 2)}

`;

    const response = await withRetry(
      () =>
        ai.models.generateContent({

          model: ANALYSIS_MODEL,

          contents: [
            ...imageParts,

            {
              text: analysisPrompt
            }
          ],

          config: {
            responseMimeType:
              "application/json"
          }
        }),

      4
    );

    const responseText =
      typeof response.text === "string"
        ? response.text
        : response.text();

    let raw;

    try {

      raw = JSON.parse(
        responseText
          .replace(/^```json/i, "")
          .replace(/^```/i, "")
          .replace(/```$/i, "")
          .trim()
      );

    } catch (error) {

      console.error(
        "Gemini returned invalid JSON:",
        responseText
      );

      return res.status(500).json({
        success: false,
        error: "Gemini returned invalid JSON"
      });
    }

    const analysis =
      normalizeAnalysis(raw);

    analysis.input.photoCount =
      photoList.length;

    return res.json({
      success: true,
      analysis
    });

  } catch (error) {

    console.error(
      "FC FACE LAB ERROR:",
      error
    );

    const status =
      error?.status ||
      error?.code ||
      error?.response?.status;

    if (
      status === 503 ||
      status === 429
    ) {

      return res.status(503).json({
        success: false,
        error:
          "Gemini is temporarily busy. Please try again shortly."
      });
    }

    return res.status(500).json({
      success: false,
      error:
        error?.message ||
        "Gemini request failed"
    });
  }
});

app.listen(PORT, () => {

  console.log(
    `FC FACE LAB backend running on port ${PORT}`
  );

});
