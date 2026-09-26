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

app.get("/", (req, res) => {
  res.json({
    ok: true,
    service: "FC FACE LAB",
    message: "Backend is running"
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

    const response = await ai.models.generateContent({
      model: "gemini-3.8-flash",

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
    });

    let analysis;

    try {
      const responseText =
        typeof response.text === "string"
          ? response.text
          : response.text();

      analysis = JSON.parse(responseText);

    } catch (error) {

      console.error("JSON parse error:", error);

      return res.status(500).json({
        success: false,
        error: "Gemini returned invalid JSON"
      });
    }

    let simulationImage = null;

    try {

      const imageResponse = await ai.models.generateContent({

        model: "gemini-3.1-flash-image",

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

      });

      const parts =
        imageResponse.candidates?.[0]?.content?.parts || [];

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

    }

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
