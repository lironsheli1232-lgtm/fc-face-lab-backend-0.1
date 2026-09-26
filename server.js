import express from "express";
import cors from "cors";
import { GoogleGenAI } from "@google/genai";

const app = express();
const PORT = process.env.PORT || 3000;

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY
});

app.use(cors());
app.use(express.json({ limit: "15mb" }));

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
        error: "GEMINI_API_KEY is missing"
      });
    }

    const { image, mimeType } = req.body;

    if (!image || !mimeType) {
      return res.status(400).json({
        error: "Image is missing"
      });
    }

    const analysisPrompt = `
Analyze the person's face in the supplied photograph.

The goal is to recreate the person's appearance as closely
as possible in EA FC 27 Create A Player.

Analyze only visible characteristics:

- face shape
- skin tone
- complexion
- eyes
- eyebrows
- nose
- mouth
- cheeks
- jaw
- chin
- ears
- hairstyle
- hair color
- facial hair

Be precise.

Do not invent official EA FC 27 slider numbers
or option names.

Return ONLY valid JSON using this exact structure:

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
      model: "gemini-2.5-flash",
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
      analysis = JSON.parse(response.text);
    } catch {
      return res.status(500).json({
        error: "Gemini returned invalid JSON",
        raw: response.text
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
Create a realistic football video-game character
based on the supplied person's photograph.

Preserve the person's:

- face shape
- eyes
- eyebrows
- nose
- mouth
- jaw
- chin
- skin tone
- hairstyle
- facial hair

Make it look like a modern football video game
Create-A-Player render.

Head and shoulders.
Neutral expression.
Realistic 3D face.
Clean background.
No text.
No logos.
`
          }
        ],
        config: {
          responseModalities: ["IMAGE"]
        }
      });

      const parts =
        imageResponse.candidates?.[0]?.content?.parts || [];

      const imagePart = parts.find(
        part => part.inlineData?.data
      );

      if (imagePart) {
        simulationImage =
          `data:${
            imagePart.inlineData.mimeType || "image/png"
          };base64,${
            imagePart.inlineData.data
          }`;
      }

    } catch (imageError) {
      console.error(
        "Simulation image failed:",
        imageError
      );
    }

    res.json({
      success: true,
      analysis: analysis,
      simulationImage: simulationImage
    });

  } catch (error) {
    console.error(error);

    res.status(500).json({
      success: false,
      error:
        error.message ||
        "Gemini request failed"
    });
  }
});

app.listen(PORT, () => {
  console.log(
    `FC FACE LAB backend running on port ${PORT}`
  );
});
