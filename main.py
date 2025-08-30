from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import pdfplumber
import docx
import json
import io

# ---------- Setup ----------
load_dotenv()

# Initialize OpenAI client with error handling
try:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    client = OpenAI(api_key=api_key)
    print("✅ OpenAI client initialized successfully")
except Exception as e:
    print(f"❌ Error initializing OpenAI client: {e}")
    raise

app = FastAPI(title="AI Assignment Generator & Evaluator", version="1.0.0")

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (for the frontend)
# Create a 'static' directory and put your HTML file there
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------- Models ----------
class AssignmentRequest(BaseModel):
    topics: Optional[List[str]] = None
    difficulty: str
    mcq: int
    subjective: int
    lecture_text: Optional[str] = None

class SubmissionRequest(BaseModel):
    submission: List[Dict[str, Any]]

class QuestionResponse(BaseModel):
    id: int
    type: str
    question: str
    answer: str

class EvaluationRequest(BaseModel):
    questions: List[Dict[str, Any]]  # Original questions with correct answers
    responses: List[QuestionResponse]  # Student responses

# ---------- Utils ----------
def extract_text_from_file(file: UploadFile) -> str:
    """Extract text from uploaded PDF, DOCX, or TXT"""
    try:
        content = ""
        file_content = file.file.read()
        
        if file.filename.endswith(".pdf"):
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                content = " ".join([page.extract_text() or "" for page in pdf.pages])
        elif file.filename.endswith(".docx"):
            doc = docx.Document(io.BytesIO(file_content))
            content = " ".join([para.text for para in doc.paragraphs])
        elif file.filename.endswith(".txt"):
            content = file_content.decode("utf-8")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF, DOCX, or TXT files.")
        
        # Reset file pointer for potential reuse
        file.file.seek(0)
        return content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# ---------- Assignment Generator ----------
def generate_assignment(topics: Optional[List[str]], difficulty: str, mcq: int, subjective: int, lecture_text: Optional[str] = None):
    try:
        # Construct prompt
        if lecture_text:
            base = f"""
            You are an expert computer vision professor.
            Create an assignment STRICTLY based on the following lecture content:

            {lecture_text[:5000]}

            Focus on the key concepts, algorithms, and techniques mentioned in this content.
            """
        else:
            topics_str = ", ".join(topics) if topics else "general computer vision concepts"
            base = f"""
            You are an expert computer vision professor.
            Create an assignment based on the following topics: {topics_str}
            
            Cover fundamental and advanced concepts in these areas.
            """

        prompt = f"""
        {base}

        Requirements:
        - Difficulty level: {difficulty} (beginner/intermediate/advanced)
        - Number of MCQs: {mcq}
        - Number of Subjective Questions: {subjective}
        - Make MCQs have exactly 4 options (A, B, C, D), only one correct.
        - Subjective questions should be conceptual and require detailed explanation.
        - Questions should be relevant to computer vision field.
        - Ensure progressive difficulty within the assignment.

        Return STRICTLY valid JSON in the following format:

        {{
          "assignment": [
            {{
              "id": 1,
              "type": "mcq",
              "question": "What is the primary purpose of edge detection in computer vision?",
              "options": ["A) Image compression", "B) Feature extraction", "C) Color enhancement", "D) Noise reduction"],
              "correct_answer": "B",
              "explanation": "Brief explanation of correct answer"
            }},
            {{
              "id": 2,
              "type": "subjective",
              "question": "Explain the difference between supervised and unsupervised learning in computer vision.",
              "max_marks": 5,
              "marking_scheme": "Key points to cover for full marks"
            }}
          ]
        }}

        Do not include any text outside the JSON structure.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )

        # Parse JSON response
        content = response.choices[0].message.content.strip()
        
        # Try to extract JSON if there's extra text
        if not content.startswith('{'):
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                content = content[start:end]
        
        return json.loads(content)
    
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response as JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating assignment: {str(e)}")

# ---------- Assignment Evaluator ----------
def evaluate_submission(questions: List[Dict[str, Any]], responses: List[Dict[str, Any]]):
    try:
        prompt = f"""
        You are an expert computer vision professor grading assignments.
        
        Original Questions with Correct Answers:
        {json.dumps(questions, indent=2)}
        
        Student Responses:
        {json.dumps(responses, indent=2)}

        Grading Instructions:
        - For MCQ questions: 5 marks if correct, 0 if wrong
        - For subjective questions: Grade out of 5 marks based on:
          * Conceptual understanding (0-2 marks)
          * Technical accuracy (0-2 marks)  
          * Completeness of explanation (0-1 mark)
        
        For each response, provide:
        - "marks": score awarded
        - "feedback": detailed explanation of grading
        - "suggestions": areas for improvement (if applicable)

        Return STRICTLY valid JSON in the following format:
        {{
          "evaluation": [
            {{
              "id": 1,
              "question_type": "mcq",
              "marks_awarded": 5,
              "max_marks": 5,
              "feedback": "Correct! Edge detection is primarily used for feature extraction...",
              "suggestions": ""
            }},
            {{
              "id": 2,
              "question_type": "subjective", 
              "marks_awarded": 3,
              "max_marks": 5,
              "feedback": "Good understanding shown but missing key concepts...",
              "suggestions": "Include discussion on labeled vs unlabeled data..."
            }}
          ],
          "total_score": 8,
          "total_possible": 10,
          "percentage": 80,
          "grade": "B+",
          "overall_remarks": "Good performance with room for improvement in theoretical concepts."
        }}

        Do not include any text outside the JSON structure.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Lower temperature for consistent grading
            max_tokens=2000
        )

        content = response.choices[0].message.content.strip()
        
        # Try to extract JSON if there's extra text
        if not content.startswith('{'):
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                content = content[start:end]
        
        return json.loads(content)
    
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse evaluation response as JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating submission: {str(e)}")

# ---------- API Endpoints ----------
@app.get("/")
async def root():
    """Serve the main frontend page"""
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {"message": "AI Assignment Generator & Evaluator API", "version": "1.0.0", "note": "Place index.html in static/ directory to serve frontend"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running", "openai_configured": bool(os.getenv("OPENAI_API_KEY"))}

@app.post("/generate_assignment")
async def generate_assignment_json(request: AssignmentRequest):
    """Generate assignment using JSON request (no file upload)"""
    # Validate inputs
    if request.difficulty not in ["beginner", "intermediate", "advanced"]:
        raise HTTPException(status_code=400, detail="Difficulty must be 'beginner', 'intermediate', or 'advanced'")
    
    if request.mcq < 0 or request.subjective < 0:
        raise HTTPException(status_code=400, detail="Number of questions cannot be negative")
    
    if request.mcq + request.subjective == 0:
        raise HTTPException(status_code=400, detail="At least one question must be specified")
    
    # Generate assignment
    result = generate_assignment(request.topics, request.difficulty, request.mcq, request.subjective, request.lecture_text)
    return result

@app.post("/generate_assignment_with_topics")
async def generate_assignment_topics_only(
    topics: str = Form(..., description="Comma-separated list of topics"),
    difficulty: str = Form(..., description="beginner, intermediate, or advanced"),
    mcq: int = Form(..., description="Number of MCQ questions"),
    subjective: int = Form(..., description="Number of subjective questions")
):
    """Generate assignment with topics only (no file upload)"""
    # Parse topics
    topics_list = [topic.strip() for topic in topics.split(",") if topic.strip()]
    
    # Validate inputs
    if difficulty not in ["beginner", "intermediate", "advanced"]:
        raise HTTPException(status_code=400, detail="Difficulty must be 'beginner', 'intermediate', or 'advanced'")
    
    if mcq < 0 or subjective < 0:
        raise HTTPException(status_code=400, detail="Number of questions cannot be negative")
    
    if mcq + subjective == 0:
        raise HTTPException(status_code=400, detail="At least one question must be specified")
    
    # Generate assignment
    result = generate_assignment(topics_list, difficulty, mcq, subjective, None)
    return result

@app.post("/generate_assignment_with_file")
async def generate_assignment_with_file(
    difficulty: str = Form(...),
    mcq: int = Form(...),
    subjective: int = Form(...),
    file: UploadFile = File(...),
    topics: Optional[str] = Form(None)
):
    """Generate assignment with uploaded file (required)"""
    # Parse topics if provided
    topics_list = None
    if topics:
        topics_list = [topic.strip() for topic in topics.split(",") if topic.strip()]
    
    # Validate inputs
    if difficulty not in ["beginner", "intermediate", "advanced"]:
        raise HTTPException(status_code=400, detail="Difficulty must be 'beginner', 'intermediate', or 'advanced'")
    
    if mcq < 0 or subjective < 0:
        raise HTTPException(status_code=400, detail="Number of questions cannot be negative")
    
    if mcq + subjective == 0:
        raise HTTPException(status_code=400, detail="At least one question must be specified")
    
    # Extract text from file
    lecture_text = extract_text_from_file(file)
    if not lecture_text:
        raise HTTPException(status_code=400, detail="Could not extract text from uploaded file")
    
    # Generate assignment
    result = generate_assignment(topics_list, difficulty, mcq, subjective, lecture_text)
    return result

@app.post("/evaluate_submission")
async def evaluate_submission_api(request: EvaluationRequest):
    """Evaluate submission using JSON request"""
    # Validate input
    if not request.questions or not request.responses:
        raise HTTPException(status_code=400, detail="Both questions and responses are required")
    
    # Convert responses to the format expected by evaluate_submission
    responses_dict = [{"id": r.id, "type": r.type, "question": r.question, "answer": r.answer} for r in request.responses]
    
    result = evaluate_submission(request.questions, responses_dict)
    return result

@app.post("/evaluate_with_json_file")
async def evaluate_with_json_file(file: UploadFile = File(...)):
    """Evaluate assignment from uploaded JSON file"""
    try:
        # Check file type
        if not file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="Please upload a JSON file")
        
        # Read and parse JSON file
        file_content = await file.read()
        try:
            data = json.loads(file_content.decode('utf-8'))
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON file")
        
        # Validate JSON structure
        if "questions" not in data or "responses" not in data:
            raise HTTPException(status_code=400, detail="JSON must contain 'questions' and 'responses' fields")
        
        # Evaluate submission
        result = evaluate_submission(data["questions"], data["responses"])
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating from file: {str(e)}")

@app.post("/evaluate_submission_simple")
async def evaluate_simple(req: SubmissionRequest):
    """Simplified evaluation endpoint for backward compatibility"""
    try:
        # Extract questions and responses from submission
        questions = []
        responses = []
        
        for item in req.submission:
            if "correct_answer" in item or "marking_scheme" in item:
                questions.append(item)
            responses.append({
                "id": item.get("id"),
                "type": item.get("type"),
                "question": item.get("question", ""),
                "answer": item.get("answer", item.get("response", ""))
            })
        
        result = evaluate_submission(questions, responses)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in simple evaluation: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Create static directory if it doesn't exist
    if not os.path.exists("static"):
        os.makedirs("static")
        print("Created 'static' directory. Place your index.html file there.")
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


       


