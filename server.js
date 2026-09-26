import express from "express";
import cors from "cors";
import { GoogleGenAI } from "@google/genai";

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: "15mb" }));

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY
});

const ANALYSIS_MODEL = "gemini-3.8-flash";
const IMAGE_MODEL = "gemini-3.1-flash-image";


// --------------------------------------------------
// Retry helper
// --------------------------------------------------

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
        error?.message || "";

      const isTemporary =
        status === 503 ||
        status === 429 ||
        message.includes("high demand") ||
        message.includes("UNAVAILABLE") ||
        message.includes("overloaded") ||
        message.includes("RESOURCE_EXHAUSTED");

      if (!isTemporary || attempt === attempts) {
        throw error;
      }

      const waitTime =
        attempt === 1 ? 2500 :
        attempt === 2 ? 5000 :
        9000;

      console.log(
        `Gemini temporarily unavailable. Retry ${attempt + 1}/${attempts} in ${waitTime}ms`
      );

      await new Promise(resolve =>
        setTimeout(resolve, waitTime)
      );
    }
  }

  throw lastError;
}


// --------------------------------------------------
// Home
// --------------------------------------------------

app.get("/", (req, res) => {

  res.json({
    ok: true,
    service: "FC FACE LAB",
    message: "Backend is running"
  });

});


// --------------------------------------------------
// Face analysis
// --------------------------------------------------

app.post("/analyze", async (req, res) => {

  try {

    if (!process.env.GEMINI_API_KEY) {

      return res.status(500).json({
        success: false,
        error: "GEMINI_API_KEY is missing"
      });

    }


    const { image, mimeType } = req.body;


    if (!image || !mimeType) {

      return res.status(400).json({
        success: false,
        error: "Image is missing"
      });

    }


    const analysisPrompt = `
Analyze the person's face in the supplied photograph.

The goal is to recreate the person's appearance as closely as possible
in EA FC 27 Create A Player.

Analyze ONLY characteristics that are actually visible in the photograph.

Analyze:

- face shape
- skin tone
- complexion
- eye shape
- eye size
- eye spacing
- eyebrow shape
- eyebrow thickness
- nose shape
- nose width
- nose length
- mouth shape
- lip thickness
- cheek structure
- jaw shape
- jaw width
- chin shape
- chin size
- ear shape and size
- hairstyle
- hair length
- hair texture
- hair color
- facial hair
- approximate age appearance

Be extremely specific and descriptive.

Do NOT invent EA FC 27 slider numbers.
Do NOT invent official EA FC 27 option names.
Do NOT claim that a feature is visible if it cannot actually be determined
from the photograph.

Return ONLY valid JSON using exactly this structure:

{
  "faceShape": "",
  "skinTone": "",
  "complexion": "",
  "eyes": "",
  "eyebrows": "",
  "nose": "",
  "mouth": "",
  "cheeksJaw": "",
  "chin": "",
  "ears": "",
  "hairStyle": "",
  "hairColor": "",
  "facialHair": "",
  "overall": ""
}
`;


    // --------------------------------------------------
    // Gemini analysis with automatic retry
    // --------------------------------------------------

    const response = await withRetry(
      () =>
        ai.models.generateContent({

          model: ANALYSIS_MODEL,

          contents: [

            {
              inlineData: {
                mimeType: mimeType,
                data: image
              }
            },

            {
              text: analysisPrompt
            }

          ],

          config: {
            responseMimeType: "application/json"
          }

        }),
      4
    );


    let analysis;


    try {

      const responseText =
        typeof response.text === "string"
          ? response.text
          : response.text();

      analysis = JSON.parse(responseText);

    } catch (error) {

      console.error(
        "JSON parse error:",
        error
      );

      return res.status(500).json({
        success: false,
        error: "Gemini returned invalid JSON"
      });

    }


    // --------------------------------------------------
    // AI visual simulation
    // --------------------------------------------------

    let simulationImage = null;


    try {

      const imageResponse = await withRetry(
        () =>
          ai.models.generateContent({

            model: IMAGE_MODEL,

            contents: [

              {
                inlineData: {
                  mimeType: mimeType,
                  data: image
                }
              },

              {
                text: `
Create a realistic football video-game Create-A-Player
visualization based on the supplied person's photograph.

Preserve the person's visible identity and facial characteristics:

- face shape
- eyes
- eyebrows
- nose
- mouth
- cheeks
- jaw
- chin
- ears
- skin tone
- hairstyle
- hair color
- facial hair

The result should look like a realistic modern football
video game character.

Head and shoulders portrait.
Neutral expression.
Front-facing or slightly angled toward camera.
Realistic 3D skin.
Realistic facial proportions.
Clean neutral background.

Do not add text.
Do not add logos.
Do not add football shirts.
Do not change the person's fundamental facial characteristics.
`
              }

            ],

            config: {
              responseModalities: ["IMAGE"]
            }

          }),
        3
      );


      const parts =
        imageResponse
          ?.candidates?.[0]
          ?.content
          ?.parts || [];


      const imagePart =
        parts.find(
          part =>
            part.inlineData &&
            part.inlineData.data
        );


      if (imagePart) {

        simulationImage =
          `data:${imagePart.inlineData.mimeType || "image/png"};base64,${imagePart.inlineData.data}`;

      }

    } catch (imageError) {

      console.error(
        "Simulation image failed:",
        imageError
      );

      // The face analysis itself can still succeed
      // even if the visual simulation fails.

    }


    // --------------------------------------------------
    // Final response
    // --------------------------------------------------

    return res.json({

      success: true,

      analysis: analysis,

      simulationImage: simulationImage

    });


  } catch (error) {

    console.error(
      "Gemini request failed:",
      error
    );


    const status =
      error?.status ||
      error?.code;


    if (status === 503 || status === 429) {

      return res.status(503).json({

        success: false,

        error:
          "Gemini is temporarily busy. The server tried several times automatically. Please try again in a moment."

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


// --------------------------------------------------
// Start server
// --------------------------------------------------

app.listen(PORT, () => {

  console.log(
    `FC FACE LAB backend running on port ${PORT}`
  );

});
