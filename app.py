import json
import ast
import random
import sqlite3
import os
import hashlib
import hmac
import secrets
import subprocess
import tempfile
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Optional features: the app still works if these packages are unavailable.
try:
    from google import genai
except ImportError:
    genai = None

try:
    from gtts import gTTS
    import io
except ImportError:
    gTTS = None
    io = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None


# ----------------------------- APP CONFIG -----------------------------
st.set_page_config(
    page_title="AI Placement Prep & Tracker",
    page_icon="🎯",
    layout="wide",
)

DB_NAME = "placement.db"
MODEL_NAME = "gemini-3.8-flash"
LEVELS = ["Easy", "Medium", "Hard"]


def score_percent(score, total):
    """Return a safe percentage score, rounded to one decimal place."""
    try:
        total_value = float(total)
        score_value = float(score)
        if total_value <= 0:
            return 0.0
        return round((score_value / total_value) * 100, 1)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


# ----------------------------- DATABASE ------------------------------
def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)


def init_db():
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS seen_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student TEXT NOT NULL,
                category TEXT NOT NULL,
                topic TEXT NOT NULL,
                question TEXT NOT NULL,
                normalized_question TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(student, category, topic, normalized_question)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student TEXT NOT NULL,
                category TEXT NOT NULL,
                topic TEXT NOT NULL,
                score REAL NOT NULL,
                total REAL NOT NULL,
                difficulty TEXT,
                notes TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_result(student, category, topic, score, total, difficulty="", notes=""):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO results
               (student, category, topic, score, total, difficulty, notes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                student,
                category,
                topic,
                float(score),
                float(total),
                difficulty,
                notes,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()


def load_results(student):
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT category, topic, score, total, difficulty, notes, created_at "
            "FROM results WHERE student = ? ORDER BY id DESC",
            conn,
            params=(student,),
        )


init_db()


# ----------------------------- QUESTION BANK -------------------------
# Add more questions to these lists whenever you want.
QUESTION_BANK = {
    "Aptitude": {
        "Percentages": {
            "Easy": [
                {
                    "q": "What is 20% of 150?",
                    "options": ["15", "20", "30", "35"],
                    "answer": "30",
                    "explanation": "20/100 × 150 = 30.",
                },
                {
                    "q": "A number increases from 50 to 60. What is the percentage increase?",
                    "options": ["10%", "15%", "20%", "25%"],
                    "answer": "20%",
                    "explanation": "(10/50) × 100 = 20%.",
                },
            ],
            "Medium": [
                {
                    "q": "A value is reduced by 20% and becomes 240. What was the original value?",
                    "options": ["280", "288", "300", "320"],
                    "answer": "300",
                    "explanation": "80% of the original is 240, so original = 240/0.8 = 300.",
                },
                {
                    "q": "A student's score rises from 64 to 80. Find the percentage increase.",
                    "options": ["20%", "25%", "16%", "30%"],
                    "answer": "25%",
                    "explanation": "Increase = 16; (16/64) × 100 = 25%.",
                },
            ],
            "Hard": [
                {
                    "q": "A price increases by 20% and then decreases by 20%. What is the net change?",
                    "options": ["No change", "4% decrease", "4% increase", "2% decrease"],
                    "answer": "4% decrease",
                    "explanation": "100 becomes 120, then 96. Net change is a 4% decrease.",
                },
                {
                    "q": "A number is increased by 25%. By what percentage must the new value decrease to return to the original?",
                    "options": ["20%", "25%", "15%", "12.5%"],
                    "answer": "20%",
                    "explanation": "100 becomes 125. Decrease 25/125 × 100 = 20%.",
                },
            ],
        },
        "Profit & Loss": {
            "Easy": [
                {
                    "q": "An item bought for ₹100 is sold for ₹120. What is the profit percentage?",
                    "options": ["10%", "15%", "20%", "25%"],
                    "answer": "20%",
                    "explanation": "Profit = ₹20; (20/100) × 100 = 20%.",
                }
            ],
            "Medium": [
                {
                    "q": "An item is sold for ₹540 at a 10% loss. Find its cost price.",
                    "options": ["₹580", "₹600", "₹620", "₹650"],
                    "answer": "₹600",
                    "explanation": "Selling price is 90% of cost price: 540/0.9 = 600.",
                }
            ],
            "Hard": [
                {
                    "q": "A shopkeeper marks an item 40% above cost and gives a 10% discount. What is the profit percentage?",
                    "options": ["20%", "24%", "26%", "30%"],
                    "answer": "26%",
                    "explanation": "If cost = 100, marked price = 140; selling price = 126. Profit = 26%.",
                }
            ],
        },
        "Time, Speed & Distance": {
            "Easy": [
                {
                    "q": "A car travels 120 km in 2 hours. What is its average speed?",
                    "options": ["40 km/h", "50 km/h", "60 km/h", "80 km/h"],
                    "answer": "60 km/h",
                    "explanation": "Speed = distance/time = 120/2 = 60 km/h.",
                }
            ],
            "Medium": [
                {
                    "q": "A 150 m train crosses a pole in 10 seconds. Find its speed.",
                    "options": ["15 m/s", "10 m/s", "20 m/s", "25 m/s"],
                    "answer": "15 m/s",
                    "explanation": "Speed = 150/10 = 15 m/s.",
                }
            ],
            "Hard": [
                {
                    "q": "A person travels equal distances at 30 km/h and 60 km/h. What is the average speed?",
                    "options": ["40 km/h", "45 km/h", "50 km/h", "30 km/h"],
                    "answer": "40 km/h",
                    "explanation": "For equal distances, average speed = 2ab/(a+b) = 40 km/h.",
                }
            ],
        },
        "Ratio & Proportion": {
            "Easy": [
                {
                    "q": "Simplify the ratio 12:18.",
                    "options": ["2:3", "3:2", "4:5", "6:9"],
                    "answer": "2:3",
                    "explanation": "Divide both values by 6.",
                }
            ],
            "Medium": [
                {
                    "q": "Divide ₹600 in the ratio 2:3. What is the larger share?",
                    "options": ["₹200", "₹240", "₹360", "₹400"],
                    "answer": "₹360",
                    "explanation": "Total parts = 5. Larger share = 3/5 × 600 = 360.",
                }
            ],
            "Hard": [
                {
                    "q": "If A:B = 3:4 and B:C = 8:5, what is A:C?",
                    "options": ["3:5", "6:5", "5:6", "4:5"],
                    "answer": "6:5",
                    "explanation": "Make B equal: A:B = 6:8 and B:C = 8:5, so A:C = 6:5.",
                }
            ],
        },
        "Averages": {
            "Easy": [
                {
                    "q": "Find the average of 4, 6, and 8.",
                    "options": ["5", "6", "7", "8"],
                    "answer": "6",
                    "explanation": "(4 + 6 + 8)/3 = 6.",
                }
            ],
            "Medium": [
                {
                    "q": "The average of 5 numbers is 12. What is their total?",
                    "options": ["50", "55", "60", "65"],
                    "answer": "60",
                    "explanation": "Total = average × count = 12 × 5 = 60.",
                }
            ],
            "Hard": [
                {
                    "q": "The average of 8 numbers is 15. If one number 9 is replaced by 17, what is the new average?",
                    "options": ["15", "16", "17", "18"],
                    "answer": "16",
                    "explanation": "Total increases by 8. New average = (120 + 8)/8 = 16.",
                }
            ],
        },
    },
    "Reasoning": {
        "Number Series": {
            "Easy": [
                {
                    "q": "Find the next number: 2, 4, 6, 8, __",
                    "options": ["9", "10", "11", "12"],
                    "answer": "10",
                    "explanation": "Each number increases by 2.",
                }
            ],
            "Medium": [
                {
                    "q": "Find the next number: 3, 6, 12, 24, __",
                    "options": ["36", "42", "48", "54"],
                    "answer": "48",
                    "explanation": "Each term is multiplied by 2.",
                }
            ],
            "Hard": [
                {
                    "q": "Find the next number: 1, 4, 9, 16, 25, __",
                    "options": ["30", "35", "36", "49"],
                    "answer": "36",
                    "explanation": "These are squares: 1², 2², 3², 4², 5², so next is 6² = 36.",
                }
            ],
        },
        "Logical Reasoning": {
            "Easy": [
                {
                    "q": "All roses are flowers. Which statement must be true?",
                    "options": [
                        "All flowers are roses",
                        "All roses are flowers",
                        "No rose is a flower",
                        "All flowers are red",
                    ],
                    "answer": "All roses are flowers",
                    "explanation": "This repeats the given statement.",
                }
            ],
            "Medium": [
                {
                    "q": "If CAT is coded as DBU, how is DOG coded using the same rule?",
                    "options": ["EPH", "CNE", "FQI", "DPI"],
                    "answer": "EPH",
                    "explanation": "Each letter moves one step forward: D→E, O→P, G→H.",
                }
            ],
            "Hard": [
                {
                    "q": "A is taller than B. C is taller than A. D is shorter than B. Who is tallest?",
                    "options": ["A", "B", "C", "D"],
                    "answer": "C",
                    "explanation": "C > A > B > D, so C is tallest.",
                }
            ],
        },
        "Odd One Out": {
            "Easy": [
                {
                    "q": "Choose the odd one out.",
                    "options": ["Apple", "Mango", "Carrot", "Banana"],
                    "answer": "Carrot",
                    "explanation": "Carrot is commonly treated as a vegetable; the others are fruits.",
                }
            ],
            "Medium": [
                {
                    "q": "Choose the odd one out.",
                    "options": ["2", "3", "5", "9"],
                    "answer": "9",
                    "explanation": "2, 3, and 5 are prime numbers; 9 is not.",
                }
            ],
            "Hard": [
                {
                    "q": "Choose the odd one out.",
                    "options": ["8", "27", "64", "100"],
                    "answer": "100",
                    "explanation": "8, 27, and 64 are perfect cubes; 100 is not.",
                }
            ],
        },
        "Coding-Decoding": {
            "Easy": [
                {
                    "q": "If A = 1, B = 2, C = 3, what is the value of CAB?",
                    "options": ["312", "123", "321", "213"],
                    "answer": "312",
                    "explanation": "C=3, A=1, B=2, so CAB = 312.",
                }
            ],
            "Medium": [
                {
                    "q": "If BOOK is coded as CPPL (each letter +1), how is PEN coded?",
                    "options": ["QFO", "ODM", "RGP", "QEN"],
                    "answer": "QFO",
                    "explanation": "P→Q, E→F, N→O.",
                }
            ],
            "Hard": [
                {
                    "q": "If MANGO is coded as NBOHP (each letter +1), how is GRAPE coded?",
                    "options": ["HSBQF", "FQZOD", "HTBQF", "HSAPF"],
                    "answer": "HSBQF",
                    "explanation": "G→H, R→S, A→B, P→Q, E→F.",
                }
            ],
        },
    },
}

INTERVIEW_QUESTIONS = {
    "HR": [
        "Tell me about yourself.",
        "Why should we hire you?",
        "What are your strengths and one area you are improving?",
        "Describe a challenge you faced in a team and how you handled it.",
        "Where do you see yourself in five years?",
    ],
    "Python": [
        "What is the difference between a list and a tuple in Python?",
        "Explain Python dictionaries and give a use case.",
        "What is the difference between == and is in Python?",
        "What are exceptions, and how do try and except work?",
        "Explain the purpose of a generator in Python.",
    ],
    "Java": [
        "Explain the four pillars of object-oriented programming.",
        "What is the difference between an interface and an abstract class?",
        "What is the difference between == and equals() in Java?",
        "Explain method overloading and method overriding.",
        "What is exception handling in Java?",
    ],
    "SQL / DBMS": [
        "What is the difference between a primary key and a foreign key?",
        "Explain the difference between WHERE and HAVING.",
        "What is normalization and why is it used?",
        "Explain INNER JOIN and LEFT JOIN.",
        "What are ACID properties in a database?",
    ],
    "C Programming": [
        "What is the difference between an array and a pointer in C?",
        "Explain the difference between malloc(), calloc(), and free().",
        "What is a segmentation fault, and what commonly causes it?",
        "What is the difference between a structure and a union in C?",
        "Explain call by value and how pointers can be used to modify a variable in a function.",
    ],
    "C++ / OOP": [
        "Explain encapsulation, inheritance, polymorphism, and abstraction with examples.",
        "What is the difference between a class and an object?",
        "What is the difference between function overloading and function overriding?",
        "What are constructors and destructors in C++?",
        "What is the difference between a virtual function and a pure virtual function?",
    ],
    "Data Structures & Algorithms": [
        "Compare arrays and linked lists, including one advantage of each.",
        "Explain how a stack differs from a queue and give one use case for each.",
        "What is the time complexity of binary search, and why does the input need to be sorted?",
        "How would you detect a cycle in a linked list?",
        "Compare time complexity for linear search and binary search.",
    ],
    "Operating Systems": [
        "What is the difference between a process and a thread?",
        "What are the four necessary conditions for deadlock?",
        "Explain paging and why virtual memory is useful.",
        "What is the difference between preemptive and non-preemptive scheduling?",
        "What is a context switch?",
    ],
    "Computer Networks": [
        "Explain the OSI model and the purpose of its layers.",
        "What is the difference between TCP and UDP?",
        "What happens when you enter a URL into a browser?",
        "Explain the difference between an IP address and a MAC address.",
        "What is DNS and why is it needed?",
    ],
    "Computer Vision / AI": [
        "What is the difference between image classification, object detection, and image segmentation?",
        "What is a convolutional neural network (CNN), and why is it useful for images?",
        "What are overfitting and underfitting, and how can you reduce overfitting?",
        "What is the purpose of a training, validation, and test split?",
        "How would you evaluate an object-detection model? Name relevant metrics.",
    ],
    "Projects & Resume": [
        "Explain one project from your resume, including the problem, your solution, and your contribution.",
        "What was the most difficult bug you faced in a project, and how did you solve it?",
        "How did you choose the technologies used in your project?",
        "If you had one more week, what would you improve in your project?",
        "How did you test your project and check that it worked correctly?",
    ],
    "Mixed Placement Interview (50 Questions)": [
        'Tell me about yourself and connect your skills to this role.',
        'Why do you want to work for our company?',
        'What makes you a good fit for this position?',
        'Describe a time you learned a new technology quickly.',
        'Tell me about a time you made a mistake and what you learned.',
        'How do you handle deadlines when several tasks are urgent?',
        'Describe a disagreement in a team and how you resolved it.',
        'What motivates you to do your best work?',
        'What is one skill you are currently improving?',
        'How would your teammates describe your working style?',
        'Explain the difference between a process and a thread.',
        'What is the difference between stack memory and heap memory?',
        'Explain the four pillars of object-oriented programming.',
        'What is the difference between compile-time and run-time errors?',
        'What is the difference between an array and a linked list?',
        'How does a hash table provide fast lookup?',
        'Explain the time complexity of binary search.',
        'What is the difference between a stack and a queue?',
        'What is recursion, and what is a base case?',
        'How would you find duplicates in an integer array?',
        'What is the difference between == and equals() in Java?',
        'What is the difference between a Python list and tuple?',
        'How does exception handling improve program reliability?',
        'What is the purpose of a constructor?',
        'Explain method overloading versus method overriding.',
        'What is a primary key, and why is it important?',
        'Explain INNER JOIN versus LEFT JOIN.',
        'What is database normalization?',
        'What is the purpose of an index in a database?',
        'Explain the ACID properties of a transaction.',
        'What is the difference between HTTP and HTTPS?',
        'What is DNS, and what happens during name resolution?',
        'Explain the OSI model briefly.',
        'What is the difference between TCP and UDP?',
        'What is deadlock, and how can it be prevented?',
        'What is virtual memory?',
        'What is the difference between supervised and unsupervised learning?',
        'What are overfitting and underfitting?',
        'Why do we split data into training and testing sets?',
        'What is the purpose of prompt engineering in generative AI?',
        'Explain one project you built and the problem it solves.',
        'What was your individual contribution to your project?',
        'How did you test your project?',
        'What was the most challenging bug you fixed?',
        'If you had more time, what would you improve in your project?',
        'How do you decide which data structure to use?',
        'How would you explain a technical concept to a non-technical person?',
        'What would you do if you did not know the answer to an interview question?',
        'How do you respond to constructive criticism?',
        'Where do you see yourself professionally in three years?',
    ],
}


# ----------------------------- HELPERS --------------------------------
def get_client(api_key):
    if not api_key or not api_key.strip():
        return None
    if genai is None:
        st.error("Gemini library is missing. Run: python -m pip install google-genai")
        return None
    try:
        return genai.Client(api_key=api_key.strip())
    except Exception as exc:
        st.error(f"Could not initialize Gemini: {exc}")
        return None


def ask_gemini(client, prompt):
    if client is None:
        return None
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        return getattr(response, "text", None) or "The AI returned an empty response."
    except Exception as exc:
        # Keep provider errors out of the UI; quota errors can be very long.
        message = str(exc).lower()
        if any(term in message for term in ("resource_exhausted", "quota", "429", "rate limit")):
            return "AI review is temporarily unavailable because the Gemini API quota has been reached. Your question and hints are still available. Try again after the quota resets, or check your Gemini API usage and limits."
        if any(term in message for term in ("503", "unavailable", "overloaded")):
            return "Gemini is temporarily busy. Please try again in a little while. Your question and hints are still available."
        return "AI review could not connect right now. Check your API key and internet connection, then try again."


def run_python_test_cases(code, test_cases, timeout_seconds=2):
    """Run the bundled Python tests and keep input, stdout, and errors separate.

    IMPORTANT: subprocess timeouts are not a security sandbox. Do not expose this
    in a public multi-user deployment without moving execution to a real isolated
    runner (for example, a sandboxed code-execution service/container).
    """
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return {
            "status": "syntax_error",
            "message": f"Syntax error on line {exc.lineno or '?'}: {exc.msg}",
        }

    with tempfile.TemporaryDirectory(prefix="placement_test_") as temp_dir:
        script_path = Path(temp_dir) / "submission.py"
        script_path.write_text(code, encoding="utf-8")
        results = []

        for case_no, (stdin_text, expected_output) in enumerate(test_cases, start=1):
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", str(script_path)],
                    input=stdin_text,
                    text=True,
                    capture_output=True,
                    timeout=timeout_seconds,
                    cwd=temp_dir,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                partial_stdout = exc.stdout or ""
                if isinstance(partial_stdout, bytes):
                    partial_stdout = partial_stdout.decode("utf-8", errors="replace")
                results.append({
                    "case": case_no,
                    "input": stdin_text,
                    "passed": False,
                    "expected": expected_output.strip(),
                    "actual": partial_stdout.strip(),
                    "error_details": f"Time limit exceeded ({timeout_seconds} seconds). Check for an infinite loop or input statement that expects more values.",
                    "returncode": None,
                })
                continue

            actual_stdout = proc.stdout.strip()
            expected = expected_output.strip()
            stderr = proc.stderr.strip()
            # Preserve the traceback instead of showing only its final line.
            # The temporary directory name is removed to keep the message readable.
            if stderr:
                stderr = stderr.replace(str(temp_dir) + os.sep, "")
                stderr = stderr.replace(str(temp_dir), "")
            passed = proc.returncode == 0 and actual_stdout == expected
            results.append({
                "case": case_no,
                "input": stdin_text,
                "passed": passed,
                "expected": expected,
                "actual": actual_stdout,
                "error_details": stderr if proc.returncode != 0 else "",
                "returncode": proc.returncode,
            })

        return {"status": "tested", "results": results}


def offline_code_review(subject, topic, level, task_title, task_description, code):
    """Provide safe, lightweight feedback when Gemini is unavailable; never execute submitted code."""
    submitted = code.strip()
    if not submitted:
        return "Write your solution first, then review it."

    if subject == "Python":
        try:
            tree = ast.parse(submitted)
        except SyntaxError as exc:
            line = exc.lineno or "unknown"
            return (f"**Local review — syntax issue found**\n\n"
                    f"Python reports a syntax error near line {line}: {exc.msg}.\n\n"
                    "**Hint:** Check indentation, brackets, colons, quotation marks, and whether each statement is complete. "
                    "Fix the syntax issue and review again. No full solution is shown.")

        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        checks = []
        if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "input" for node in ast.walk(tree)):
            checks.append("Input is being read; compare the number of input values and their format with the problem statement.")
        if not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print" for node in ast.walk(tree)):
            checks.append("I couldn't find a print() call. Check whether the expected output is produced.")
        if "swap" in (task_title + " " + task_description).lower() or "swapping" in (task_title + " " + task_description).lower():
            if not any(isinstance(node, ast.Name) and node.id in names for node in ast.walk(tree)):
                checks.append("Check that both input values are stored and exchanged correctly.")
        if not checks:
            checks.append("The Python syntax is valid. Now check the algorithm against the exact requirements and sample input/output.")
        return ("**Local review (Gemini is unavailable)**\n\n"
                "Python syntax looks valid. I haven't executed your program, so correctness is not confirmed.\n\n"
                "**Please check:**\n" + "\n".join(f"- {item}" for item in checks) +
                "\n\nTry a normal case and at least two edge cases. This review gives hints only, not a complete solution.")

    # Conservative checks for non-Python submissions; do not execute code or SQL.
    hints = []
    if submitted.count("(") != submitted.count(")"):
        hints.append("Check that every opening parenthesis has a matching closing parenthesis.")
    if submitted.count("{") != submitted.count("}"):
        hints.append("Check that every opening brace has a matching closing brace.")
    if subject == "SQL":
        if not any(word in submitted.lower().split() for word in ("select", "insert", "update", "delete")):
            hints.append("Check that your query begins with the SQL operation required by the task.")
        hints.append("Check table/column names, join conditions, NULL values, and whether the result matches the requested output.")
    if not hints:
        hints.append("Check the required input/output format, boundary cases, and whether each condition matches the problem statement.")
    return ("**Local review (Gemini is unavailable)**\n\n"
            "I can't confirm full correctness without a language-specific compiler or test runner.\n\n"
            "**Hints to check:**\n" + "\n".join(f"- {item}" for item in hints) +
            "\n\nYour code was not executed, and no full solution is shown.")


def offline_interview_evaluation(category, question, answer):
    """Basic local feedback so interview practice remains usable without Gemini."""
    text = answer.strip()
    words = text.split()
    word_count = len(words)
    lower = text.lower()

    score = 2
    if word_count >= 15:
        score += 2
    if word_count >= 35:
        score += 1
    if any(term in lower for term in ("because", "for example", "for instance", "therefore", "first", "finally")):
        score += 1
    if any(term in lower for term in ("project", "implemented", "developed", "team", "result", "experience")):
        score += 1
    if any(term in lower for term in ("however", "trade-off", "complexity", "advantage", "disadvantage", "database", "class", "object", "exception", "query", "memory", "algorithm")):
        score += 1
    score = max(1, min(7, score))

    strengths = []
    improvements = []
    if word_count >= 25:
        strengths.append("Your answer includes enough detail to evaluate.")
    else:
        improvements.append("Expand your answer with a clear explanation and one specific example.")
    if any(term in lower for term in ("for example", "for instance", "project", "experience")):
        strengths.append("You included an example or practical context.")
    else:
        improvements.append("Add a real project, coursework, or practical example where relevant.")
    if any(term in lower for term in ("first", "second", "finally", "because", "therefore")):
        strengths.append("Your answer uses reasoning or structure words.")
    else:
        improvements.append("Organize the answer into 2–3 clear points.")

    if category == "HR":
        sample = "I am a computer science student who enjoys solving problems and learning new technologies. In my academic work, I have practiced programming and worked on projects that helped me improve my technical and teamwork skills. I am looking for an opportunity to contribute, learn from the team, and grow as a software professional."
    else:
        sample = "Start with a precise definition, explain how it works, compare it with a related concept if useful, and finish with a small example from code or a project."

    return {
        "score": score,
        "feedback": "Offline practice feedback (not an AI-generated correctness check): " +
                    ("Your answer has a useful starting point. Keep it focused and support key claims with examples." if word_count >= 15 else "Your answer is too short to demonstrate your understanding yet. Add more explanation and an example."),
        "strengths": strengths or ["You attempted the question."],
        "improvements": improvements or ["Check technical accuracy and include a concise example."],
        "sample_answer": sample,
        "follow_up": "Can you explain that with a specific example?",
    }


def difficulty_after_round(score, total, current_level):
    """80%+ moves up; below 50% moves down; otherwise keep current level."""
    if total <= 0:
        return current_level
    accuracy = (score / total) * 100
    idx = LEVELS.index(current_level)
    if accuracy >= 80 and idx < len(LEVELS) - 1:
        return LEVELS[idx + 1]
    if accuracy < 50 and idx > 0:
        return LEVELS[idx - 1]
    return current_level


# Expanded quantitative aptitude syllabus. Existing detailed generators are retained;
# these additional topics are generated locally so practice works without Gemini.
EXTRA_APTITUDE_TOPICS = {
    "Number System": {"Easy": [], "Medium": [], "Hard": []},
    "HCF & LCM": {"Easy": [], "Medium": [], "Hard": []},
    "Simplification & Approximation": {"Easy": [], "Medium": [], "Hard": []},
    "Simple & Compound Interest": {"Easy": [], "Medium": [], "Hard": []},
    "Mixtures & Alligation": {"Easy": [], "Medium": [], "Hard": []},
    "Time & Work": {"Easy": [], "Medium": [], "Hard": []},
    "Pipes & Cisterns": {"Easy": [], "Medium": [], "Hard": []},
    "Algebra & Equations": {"Easy": [], "Medium": [], "Hard": []},
    "Progressions & Series": {"Easy": [], "Medium": [], "Hard": []},
    "Geometry & Mensuration": {"Easy": [], "Medium": [], "Hard": []},
    "Permutation & Combination": {"Easy": [], "Medium": [], "Hard": []},
    "Probability": {"Easy": [], "Medium": [], "Hard": []},
    "Set Theory & Venn Diagrams": {"Easy": [], "Medium": [], "Hard": []},
    "Logarithms & Exponents": {"Easy": [], "Medium": [], "Hard": []},
    "Problems on Ages": {"Easy": [], "Medium": [], "Hard": []},
    "Clocks & Calendars": {"Easy": [], "Medium": [], "Hard": []},
    "Data Interpretation": {"Easy": [], "Medium": [], "Hard": []},
    "Commercial Mathematics": {"Easy": [], "Medium": [], "Hard": []},
}

# Additional reasoning concepts. Questions are generated locally so practice continues
# even when the Gemini API quota is exhausted.
EXTRA_REASONING_TOPICS = {
    "Analogy": {level: [] for level in LEVELS},
    "Alphabet Series": {level: [] for level in LEVELS},
    "Coding-Decoding (Advanced)": {level: [] for level in LEVELS},
    "Blood Relations": {level: [] for level in LEVELS},
    "Direction Sense": {level: [] for level in LEVELS},
    "Seating Arrangement": {level: [] for level in LEVELS},
    "Ranking & Order": {level: [] for level in LEVELS},
    "Syllogisms": {level: [] for level in LEVELS},
    "Statement & Conclusion": {level: [] for level in LEVELS},
    "Calendar & Clock Reasoning": {level: [] for level in LEVELS},
    "Mathematical Operations": {level: [] for level in LEVELS},
    "Data Sufficiency": {level: [] for level in LEVELS},
}
QUESTION_BANK["Reasoning"].update(EXTRA_REASONING_TOPICS)

QUESTION_BANK["Aptitude"].update(EXTRA_APTITUDE_TOPICS)


def question_topics(category):
    return list(QUESTION_BANK[category].keys())


def normalize_question(text):
    """Normalize whitespace and case so exact duplicate questions are filtered."""
    return " ".join(str(text).split()).casefold().strip()




def load_seen_questions(student, category, topic):
    """Load previously shown questions from SQLite so restarts do not reset history."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT question FROM seen_questions WHERE student = ? AND category = ? AND topic = ? ORDER BY id",
            (student, category, topic),
        ).fetchall()
    return [row[0] for row in rows]


def save_seen_questions(student, category, topic, questions):
    """Persist question history; duplicate inserts are ignored."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [
        (student, category, topic, str(q).strip(), normalize_question(q), now)
        for q in questions if str(q).strip()
    ]
    if not rows:
        return
    with get_connection() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO seen_questions "
            "(student, category, topic, question, normalized_question, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()


def parse_json_array(text):
    """Extract a JSON array from an AI response, including fenced responses."""
    if not text:
        return []
    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]).strip()
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start < 0 or end < start:
        return []
    try:
        data = json.loads(cleaned[start:end + 1])
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def make_local_aptitude_question(topic, level):
    """Create varied offline aptitude MCQs using different problem-solving patterns."""
    import math

    difficulty = {"Easy": 1, "Medium": 2, "Hard": 3}.get(level, 1)

    def mcq(q, answer, distractors, explanation, formula):
        answer = str(answer)
        options = [answer]
        for value in distractors:
            value = str(value)
            if value not in options:
                options.append(value)
            if len(options) == 4:
                break
        # Always provide four unique options, even if a pattern has limited distractors.
        suffix = 1
        while len(options) < 4:
            candidate = f"Option {suffix}"
            suffix += 1
            if candidate not in options:
                options.append(candidate)
        random.shuffle(options)
        return {"q": q, "options": options, "answer": answer,
                "explanation": explanation, "formula": formula}

    def pct_text(value):
        return f"{value:g}%"

    if topic == "Percentages":
        pattern = _next_reasoning_mode("Aptitude:Percentages:" + level,
            (["find_part", "find_percent", "increase_decrease", "reverse", "successive"] if difficulty == 1 else
             ["find_percent", "increase_decrease", "reverse", "successive", "compare"] if difficulty == 2 else
             ["reverse", "successive", "compare", "population", "marks"]))
        if pattern == "find_part":
            p = random.choice([5, 10, 12, 15, 20, 25, 30, 40, 50, 75])
            base = random.choice([80, 100, 120, 160, 200, 240, 300, 400, 600, 800])
            ans = p * base / 100
            return mcq(f"Find {p}% of {base}.", f"{ans:g}", [f"{ans+p:g}", f"{ans-5:g}", f"{base/p:g}"],
                       f"{p}/100 × {base} = {ans:g}.", "Percentage value = (Percentage ÷ 100) × Total")
        if pattern == "find_percent":
            base = random.choice([40, 50, 60, 80, 100, 120, 150, 200, 240, 300])
            p = random.choice([10, 15, 20, 25, 30, 40, 60])
            part = base * p / 100
            return mcq(f"{part:g} is what percentage of {base}?", pct_text(p),
                       [pct_text(p+5), pct_text(max(1, p-5)), pct_text(p*2)],
                       f"({part:g} ÷ {base}) × 100 = {p}%.", "Percentage = (Part ÷ Whole) × 100")
        if pattern == "increase_decrease":
            base = random.choice([80, 100, 120, 160, 200, 240, 400, 500])
            p = random.choice([10, 15, 20, 25, 30, 40])
            up = random.choice([True, False])
            new = base * (1 + p/100) if up else base * (1 - p/100)
            word = "increases" if up else "decreases"
            change_word = "increase" if up else "decrease"
            return mcq(f"A value of {base} {word} to {new:g}. Find the percentage {change_word}.", pct_text(p),
                       [pct_text(p+10), pct_text(max(1,p-5)), pct_text(p*2)],
                       f"Change = {abs(new-base):g}; ({abs(new-base):g} ÷ {base}) × 100 = {p}%.",
                       "Percentage change = (Change ÷ Original value) × 100")
        if pattern == "reverse":
            p = random.choice([10, 20, 25, 30, 40])
            original = random.choice([100, 150, 200, 250, 300, 400, 500, 600])
            increased = random.choice([True, False])
            final = original * (1 + p/100) if increased else original * (1 - p/100)
            verb = "increased" if increased else "reduced"
            return mcq(f"After a value is {verb} by {p}%, it becomes {final:g}. What was the original value?",
                       f"{original:g}", [f"{final:g}", f"{original+p:g}", f"{original-p:g}"],
                       f"Original = {final:g} ÷ {(1+p/100) if increased else (1-p/100):g} = {original:g}.",
                       "Original = Final value ÷ (1 ± Percentage/100)")
        if pattern == "successive":
            base = random.choice([100, 200, 400, 500, 800, 1000])
            p, r = random.choice([10, 20, 25, 30]), random.choice([5, 10, 15, 20, 25])
            up_first = random.choice([True, False])
            factor1 = 1+p/100 if up_first else 1-p/100
            factor2 = 1-r/100 if up_first else 1+r/100
            final = base*factor1*factor2
            net = (final/base-1)*100
            ans = f"{abs(net):g}% {'increase' if net > 0 else 'decrease' if net < 0 else 'no change'}"
            return mcq(f"A value of {base} changes by {p}% {'up' if up_first else 'down'}, then by {r}% {'down' if up_first else 'up'}. Find the net change.", ans,
                       [f"{p+r}% increase", f"{abs(p-r)}% decrease", f"{p}% {'increase' if not up_first else 'decrease'}"],
                       f"Final value = {base} × {factor1:g} × {factor2:g} = {final:g}; net change = {net:g}%.",
                       "For successive changes, multiply the change factors")
        if pattern == "compare":
            a = random.choice([120, 150, 200, 240, 300, 400])
            p = random.choice([10, 20, 25, 40])
            b = a*(1+p/100)
            return mcq(f"A is {p}% greater than B. If B = {a}, what is A?", f"{b:g}",
                       [f"{a+p:g}", f"{a*(1-p/100):g}", f"{a+p*2:g}"],
                       f"A = {a} × (1 + {p}/100) = {b:g}.", "New value = Original × (1 + Percentage/100)")
        if pattern == "population":
            pop = random.choice([2000, 4000, 5000, 8000, 10000, 12000])
            p = random.choice([5, 10, 15, 20])
            change = pop*p/100
            return mcq(f"A town has {pop} people. Its population increases by {p}%. What is the new population?",
                       f"{pop+change:g}", [f"{pop-change:g}", f"{pop+2*change:g}", f"{change:g}"],
                       f"Increase = {pop} × {p}/100 = {change:g}; new population = {pop+change:g}.", "New total = Original total + Percentage increase")
        total = random.choice([50, 60, 80, 100, 120, 200])
        score = random.randint(int(total*.45), int(total*.85))
        p = score/total*100
        return mcq(f"A student scores {score} out of {total} marks. What percentage did the student score?", pct_text(p),
                   [pct_text(p+10), pct_text(max(0,p-10)), pct_text(100-p)],
                   f"({score} ÷ {total}) × 100 = {p:g}%.", "Percentage score = (Marks obtained ÷ Total marks) × 100")

    if topic == "Profit & Loss":
        patterns = ["profit_pct", "loss_pct", "find_sp", "find_cp"] if difficulty == 1 else ["profit_pct", "loss_pct", "find_sp", "find_cp", "discount"] if difficulty == 2 else ["markup_discount", "false_weight", "find_cp", "successive_discount", "profit_pct"]
        pattern = _next_reasoning_mode("Aptitude:" + topic + ":" + level, patterns)
        cp = random.choice([100, 200, 250, 300, 400, 500, 600, 800, 1000])
        p = random.choice([10, 15, 20, 25, 30, 40])
        if pattern == "profit_pct":
            sp = cp*(1+p/100)
            return mcq(f"An item costs ₹{cp} and is sold for ₹{sp:g}. Find the profit percentage.", pct_text(p),
                       [pct_text(p+5), pct_text(max(1,p-5)), pct_text(p*2)],
                       f"Profit = ₹{sp-cp:g}; profit % = ({sp-cp:g}/{cp}) × 100 = {p}%.", "Profit % = (Profit ÷ Cost Price) × 100")
        if pattern == "loss_pct":
            loss = random.choice([5, 10, 15, 20, 25, 30])
            sp = cp*(1-loss/100)
            return mcq(f"An item bought for ₹{cp} is sold for ₹{sp:g}. Find the loss percentage.", pct_text(loss),
                       [pct_text(loss+5), pct_text(max(1,loss-5)), pct_text(loss*2)],
                       f"Loss = ₹{cp-sp:g}; loss % = ({cp-sp:g}/{cp}) × 100 = {loss}%.", "Loss % = (Loss ÷ Cost Price) × 100")
        if pattern == "find_sp":
            sp = cp*(1+p/100)
            return mcq(f"The cost price of an item is ₹{cp}. Find its selling price at a profit of {p}%.", f"₹{sp:g}",
                       [f"₹{cp*(1-p/100):g}", f"₹{cp+p:g}", f"₹{cp*(1+(p+10)/100):g}"],
                       f"Selling price = {cp} × (1 + {p}/100) = ₹{sp:g}.", "SP = CP × (100 + Profit%)/100")
        if pattern == "find_cp":
            loss = random.choice([10, 20, 25, 30])
            sp = cp*(1-loss/100)
            return mcq(f"An item is sold for ₹{sp:g} at a loss of {loss}%. Find its cost price.", f"₹{cp:g}",
                       [f"₹{sp:g}", f"₹{cp+50:g}", f"₹{cp-50:g}"],
                       f"SP is {100-loss}% of CP, so CP = {sp:g} ÷ {(100-loss)/100:g} = ₹{cp:g}.", "CP = SP ÷ (1 − Loss%/100)")
        if pattern == "discount":
            mp = random.choice([500, 600, 800, 1000, 1200, 1500])
            d = random.choice([10, 15, 20, 25, 30])
            sp = mp*(1-d/100)
            return mcq(f"An item marked at ₹{mp} is sold with a {d}% discount. Find the selling price.", f"₹{sp:g}",
                       [f"₹{mp*d/100:g}", f"₹{mp*(1+d/100):g}", f"₹{mp-d:g}"],
                       f"Discount = ₹{mp*d/100:g}; SP = ₹{mp} − ₹{mp*d/100:g} = ₹{sp:g}.", "SP = Marked Price × (100 − Discount%)/100")
        if pattern == "markup_discount":
            markup, disc = random.choice([(20,10),(25,10),(30,15),(40,10),(50,20),(60,25)])
            mp = cp*(1+markup/100); sp = mp*(1-disc/100); net = (sp/cp-1)*100
            ans = f"{abs(net):g}% {'profit' if net >= 0 else 'loss'}"
            return mcq(f"An item costs ₹{cp}, is marked {markup}% above cost, then discounted by {disc}%. Find the net result.", ans,
                       [f"{markup-disc}% profit", f"{disc}% loss", f"{markup}% profit"],
                       f"MP = ₹{mp:g}; SP = ₹{sp:g}; net profit = ₹{sp-cp:g}, which is {net:g}% of CP.", "SP = CP × (1 + Markup%/100) × (1 − Discount%/100)")
        if pattern == "false_weight":
            actual = random.choice([900, 800, 750, 500])
            profit = (1000/actual-1)*100
            ans = f"{profit:g}%"
            return mcq(f"A seller charges the price of 1 kg but gives only {actual} g. What is the gain percentage, assuming cost and selling price per gram are otherwise equal?", ans,
                       [f"{100-actual/10:g}%", f"{profit/2:g}%", f"{(1000-actual)/1000*100:g}%"],
                       f"Gain % = ((1000 − {actual}) ÷ {actual}) × 100 = {profit:g}%.", "Gain % on short weight = (Shortage ÷ Actual quantity given) × 100")
        mp = random.choice([800, 1000, 1200, 1500])
        d1, d2 = random.choice([(10,10),(10,20),(20,15),(25,10)])
        sp = mp*(1-d1/100)*(1-d2/100)
        return mcq(f"A ₹{mp} item receives successive discounts of {d1}% and {d2}%. Find the final selling price.", f"₹{sp:g}",
                   [f"₹{mp*(1-(d1+d2)/100):g}", f"₹{mp*(1+d1/100):g}", f"₹{mp-sp:g}"],
                   f"Final price = {mp} × (1 − {d1}/100) × (1 − {d2}/100) = ₹{sp:g}.", "Successive discounts are applied one after another")

    if topic == "Time, Speed & Distance":
        patterns = ["distance", "speed", "time"] if difficulty == 1 else ["speed", "time", "train", "relative"] if difficulty == 2 else ["average_equal", "train_platform", "relative", "boat"]
        pattern = _next_reasoning_mode("Aptitude:" + topic + ":" + level, patterns)
        if pattern == "distance":
            speed = random.choice([25, 30, 40, 45, 50, 60, 70, 80]); hours = random.choice([2, 3, 4, 5, 6]); dist=speed*hours
            return mcq(f"A bus travels at {speed} km/h for {hours} hours. Find the distance covered.", f"{dist} km",
                       [f"{dist+speed} km", f"{dist-speed} km", f"{speed+hours} km"], f"Distance = Speed × Time = {speed} × {hours} = {dist} km.", "Distance = Speed × Time")
        if pattern == "speed":
            time = random.choice([2, 3, 4, 5, 6]); speed = random.choice([24, 30, 36, 40, 45, 50, 60]); dist=speed*time
            return mcq(f"A car covers {dist} km in {time} hours. Find its average speed.", f"{speed} km/h",
                       [f"{speed+10} km/h", f"{max(1,speed-10)} km/h", f"{dist+time} km/h"], f"Speed = Distance ÷ Time = {dist} ÷ {time} = {speed} km/h.", "Speed = Distance ÷ Time")
        if pattern == "time":
            speed = random.choice([30, 40, 50, 60, 75, 80]); dist = speed*random.choice([2,3,4,5])
            hours = dist/speed
            return mcq(f"How long will a vehicle take to cover {dist} km at {speed} km/h?", f"{hours:g} hours",
                       [f"{hours+1:g} hours", f"{hours*2:g} hours", f"{max(.5,hours-0.5):g} hours"], f"Time = Distance ÷ Speed = {dist} ÷ {speed} = {hours:g} hours.", "Time = Distance ÷ Speed")
        if pattern == "train":
            length = random.choice([100, 120, 150, 180, 200, 240]); seconds = random.choice([8, 10, 12, 15, 20]); speed = length/seconds*3.6
            return mcq(f"A {length} m train crosses a pole in {seconds} seconds. Find its speed in km/h.", f"{speed:g} km/h",
                       [f"{length/seconds:g} km/h", f"{speed+3.6:g} km/h", f"{speed-3.6:g} km/h"], f"Speed = {length}/{seconds} = {length/seconds:g} m/s = {speed:g} km/h.", "Speed in km/h = (Distance in metres ÷ Time in seconds) × 3.6")
        if pattern == "relative":
            a, b = random.choice([(40,30),(50,35),(60,40),(45,25),(70,50)])
            same = random.choice([True, False]); rel = a-b if same else a+b
            return mcq(f"Two vehicles travel at {a} km/h and {b} km/h. What is their relative speed when moving {'in the same direction' if same else 'in opposite directions'}?", f"{rel} km/h",
                       [f"{a+b if same else a-b} km/h", f"{a} km/h", f"{b} km/h"], f"Relative speed = {'difference' if same else 'sum'} of speeds = {rel} km/h.", "Same direction: difference; opposite directions: sum")
        if pattern == "average_equal":
            a,b = random.choice([(30,60),(40,80),(25,50),(20,40),(30,90),(36,72),(45,90)])
            avg=2*a*b/(a+b)
            return mcq(f"A driver travels equal distances at {a} km/h and {b} km/h. Find the average speed.", f"{avg:g} km/h",
                       [f"{(a+b)/2:g} km/h", f"{a} km/h", f"{b} km/h"], f"Average speed = 2ab/(a+b) = 2×{a}×{b}/({a}+{b}) = {avg:g} km/h.", "For equal distances, average speed = 2ab ÷ (a+b)")
        if pattern == "train_platform":
            train = random.choice([120, 150, 180, 200]); platform = random.choice([100, 120, 150, 200]); seconds=random.choice([10,12,15,20]); speed=(train+platform)/seconds
            return mcq(f"A {train} m train crosses a {platform} m platform in {seconds} seconds. Find its speed in m/s.", f"{speed:g} m/s",
                       [f"{train/seconds:g} m/s", f"{platform/seconds:g} m/s", f"{speed+2:g} m/s"], f"Total distance = {train}+{platform} = {train+platform} m; speed = {train+platform}/{seconds} = {speed:g} m/s.", "Train speed = (Train length + Platform length) ÷ Time")
        boat = random.choice([8, 10, 12, 15]); stream = random.choice([2, 3, 4]); up=boat-stream; down=boat+stream
        return mcq(f"A boat's speed in still water is {boat} km/h and stream speed is {stream} km/h. Find its upstream speed.", f"{up} km/h",
                   [f"{down} km/h", f"{boat} km/h", f"{stream} km/h"], f"Upstream speed = {boat} − {stream} = {up} km/h.", "Upstream speed = Boat speed − Stream speed")

    if topic == "Ratio & Proportion":
        patterns = ["simplify", "missing", "divide"] if difficulty == 1 else ["divide", "combine", "direct", "ages"] if difficulty == 2 else ["combine", "inverse", "mixture", "ages"]
        pattern = _next_reasoning_mode("Aptitude:" + topic + ":" + level, patterns)
        if pattern == "simplify":
            a=random.randint(2,15); b=a*random.randint(2,9); g=math.gcd(a,b); ans=f"{a//g}:{b//g}"
            return mcq(f"Simplify the ratio {a}:{b}.", ans, [f"{b//g}:{a//g}", f"{a}:{b+1}", f"{a+1}:{b}"], f"Divide both terms by their GCD ({g}) to get {ans}.", "Divide both terms by their greatest common divisor")
        if pattern == "missing":
            a,b=random.choice([(2,3),(3,4),(4,5),(5,8),(3,7),(5,6)]); mult=random.randint(2,12); c=a*mult; d=b*mult
            return mcq(f"If {a}:{b} = {c}:x, find x.", str(d), [str(d+a), str(max(1,d-a)), str(c+b)], f"a/b = c/x, so x = ({b} × {c})/{a} = {d}.", "For a:b = c:d, cross-multiply: a×d = b×c")
        if pattern == "divide":
            a,b=random.choice([(2,3),(3,4),(4,5),(5,7),(3,5),(1,2)]); total=random.choice([240,300,360,480,600,720,900]); total += (a+b-total%(a+b))%(a+b); share=total*b//(a+b)
            return mcq(f"Divide ₹{total} in the ratio {a}:{b}. What is the larger share?", f"₹{share}", [f"₹{total-share}", f"₹{share+total//10}", f"₹{max(1,share-total//10)}"], f"Total parts = {a+b}; larger share = {b}/{a+b} × ₹{total} = ₹{share}.", "Share = (Required ratio part ÷ Sum of parts) × Total")
        if pattern == "combine":
            a,b=random.choice([(2,3),(3,4),(4,5),(5,7),(2,5)]); c,d=random.choice([(2,5),(3,7),(4,9),(5,8),(3,8)])
            common=math.lcm(b,c); A=a*(common//b); C=d*(common//c); g=math.gcd(A,C); ans=f"{A//g}:{C//g}"
            return mcq(f"If A:B = {a}:{b} and B:C = {c}:{d}, find A:C.", ans, [f"{C//g}:{A//g}", f"{a}:{d}", f"{A}:{C+1}"], f"Make B equal to {common}; A:B = {A}:{common} and B:C = {common}:{C}. Thus A:C = {ans}.", "Equalize the common term before combining ratios")
        if pattern == "direct":
            units=random.choice([3,4,5,6,8,10]); cost=units*random.choice([12,15,20,25,30]); new_units=units+random.choice([2,3,4,5]); result=cost*new_units/units
            return mcq(f"If {units} notebooks cost ₹{cost}, how much will {new_units} notebooks cost at the same rate?", f"₹{result:g}", [f"₹{cost+new_units:g}", f"₹{cost*units/new_units:g}", f"₹{result+units*5:g}"], f"Cost per notebook = ₹{cost}/{units}; total = {new_units} × ₹{cost/units:g} = ₹{result:g}.", "Direct proportion: new quantity × unit rate")
        if pattern == "ages":
            ratio_a,ratio_b=random.choice([(2,3),(3,4),(4,5),(5,7)]); total=random.choice([30,35,40,45,48,60,72]); total += (ratio_a+ratio_b-total%(ratio_a+ratio_b))%(ratio_a+ratio_b); age=total*ratio_a//(ratio_a+ratio_b)
            return mcq(f"Two people's ages are in the ratio {ratio_a}:{ratio_b}. Their total age is {total} years. Find the younger person's age.", f"{age} years", [f"{total-age} years", f"{age+5} years", f"{max(1,age-5)} years"], f"Younger age = {ratio_a}/({ratio_a+ratio_b}) × {total} = {age} years.", "Age = (Person's ratio part ÷ Total ratio parts) × Total")
        if pattern == "inverse":
            workers=random.choice([4,5,6,8,10,12]); days=random.choice([12,15,18,20,24,30]); new_workers=workers*random.choice([2,3]); new_days=workers*days/new_workers
            return mcq(f"{workers} workers finish a job in {days} days. How many days will {new_workers} workers take at the same rate?", f"{new_days:g} days", [f"{days*2:g} days", f"{days:g} days", f"{new_days+2:g} days"], f"Workers × days stays constant: {workers}×{days} = {new_workers}×x, so x = {new_days:g} days.", "Inverse proportion: workers₁ × days₁ = workers₂ × days₂")
        # Mixture ratio pattern
        a,b=random.choice([(1,2),(2,3),(3,5),(2,5),(3,4)]); total=random.choice([300,400,500,600,800,1000]); total += (a+b-total%(a+b))%(a+b); part=total*a//(a+b)
        return mcq(f"A mixture contains milk and water in the ratio {a}:{b}. If the mixture is {total} ml, how much milk is present?", f"{part} ml", [f"{total-part} ml", f"{part+50} ml", f"{max(0,part-50)} ml"], f"Milk = {a}/({a+b}) × {total} = {part} ml.", "Component quantity = Component ratio part ÷ Total ratio parts × Total mixture")

    if topic == "Averages":
        patterns = ["list", "total", "missing"] if difficulty == 1 else ["total", "missing", "replacement", "combined"] if difficulty == 2 else ["replacement", "combined", "consecutive", "target"]
        pattern = _next_reasoning_mode("Aptitude:" + topic + ":" + level, patterns)
        if pattern == "list":
            nums=[random.randint(4,30) for _ in range(3)]; total=sum(nums); avg=total/3
            ans=f"{avg:g}"
            return mcq(f"Find the average of {nums[0]}, {nums[1]}, and {nums[2]}.", ans, [f"{avg+2:g}", f"{max(0,avg-2):g}", f"{total:g}"], f"Average = ({' + '.join(map(str,nums))}) ÷ 3 = {avg:g}.", "Average = Sum of values ÷ Number of values")
        if pattern == "total":
            n=random.choice([4,5,6,8,10,12]); avg=random.choice([12,15,18,20,22,25,30,35]); total=n*avg
            return mcq(f"The average score of {n} students is {avg}. What is their total score?", str(total), [str(total-n), str(total+n), str(total+10)], f"Total = Average × Count = {avg} × {n} = {total}.", "Total = Average × Number of values")
        if pattern == "missing":
            nums=[random.randint(10,40) for _ in range(3)]; target_avg=random.choice([20,22,24,25,26,28,30]); missing=target_avg*4-sum(nums)
            if missing < 1: missing=target_avg+5; target_avg=(sum(nums)+missing)/4
            return mcq(f"The average of {nums[0]}, {nums[1]}, {nums[2]}, and x is {target_avg:g}. Find x.", f"{missing:g}", [f"{missing+4:g}", f"{max(1,missing-4):g}", f"{target_avg:g}"], f"Required total = 4 × {target_avg:g} = {4*target_avg:g}; x = {4*target_avg:g} − ({nums[0]}+{nums[1]}+{nums[2]}) = {missing:g}.", "Missing value = Required total − Sum of known values")
        if pattern == "replacement":
            n=random.choice([5,6,8,10]); avg=random.choice([20,22,25,28,30,32]); old=random.choice([10,12,15,18]); new=old+random.choice([5,10,12,15]); new_avg=avg+(new-old)/n
            return mcq(f"The average of {n} numbers is {avg}. If {old} is replaced by {new}, what is the new average?", f"{new_avg:g}", [f"{avg:g}", f"{new_avg+1:g}", f"{new_avg-1:g}"], f"Total increases by {new-old}; new average = {avg} + ({new-old}/{n}) = {new_avg:g}.", "New average = Old average + (Change in total ÷ Count)")
        if pattern == "combined":
            n1,n2=random.choice([(20,30),(15,25),(10,20),(25,35)]); a1=random.choice([60,65,70,75]); a2=a1+random.choice([5,10,15]); combined=(n1*a1+n2*a2)/(n1+n2)
            return mcq(f"Group A has {n1} students with average {a1}; Group B has {n2} students with average {a2}. Find the combined average.", f"{combined:g}", [f"{(a1+a2)/2:g}", f"{combined+5:g}", f"{combined-5:g}"], f"Combined average = ({n1}×{a1} + {n2}×{a2})/({n1}+{n2}) = {combined:g}.", "Combined average = Combined total ÷ Combined count")
        if pattern == "consecutive":
            start=random.randint(5,40); n=random.choice([3,5,7]); nums=list(range(start,start+n)); avg=sum(nums)/n
            return mcq(f"Find the average of {n} consecutive integers starting from {start}.", f"{avg:g}", [f"{avg+1:g}", f"{avg-1:g}", f"{sum(nums):g}"], f"For consecutive integers, average is the middle value: {avg:g}.", "Average of consecutive integers = (First + Last) ÷ 2")
        n=random.choice([5,6,8,10]); avg=random.choice([18,20,22,24,25,30]); desired=avg+random.choice([2,3,4,5]); needed_total=n*desired; current_total=n*avg
        return mcq(f"The average of {n} numbers is {avg}. By how much must their total increase to make the average {desired}?", str(needed_total-current_total),
                   [str(desired-avg), str((needed_total-current_total)+n), str(max(1,(needed_total-current_total)-n))],
                   f"Required increase = {n}×{desired} − {n}×{avg} = {needed_total-current_total}.", "Change in total = Number of values × Change in average")

    # Additional quantitative aptitude topic generators (offline, randomized).
    if topic in {"Number System", "HCF & LCM", "Simplification & Approximation"}:
        a, b = random.randint(12, 96), random.randint(8, 72)
        if topic == "Number System":
            mode = _next_reasoning_mode("Aptitude:" + topic + ":" + level, ["remainder", "odd_even", "divisors", "multiple"])
            if mode == "remainder":
                divisor = random.randint(3, 12); number = random.randint(20, 150); ans = number % divisor
                return mcq(f"What is the remainder when {number} is divided by {divisor}?", str(ans), [str((ans+1)%divisor), str((ans+2)%divisor), str(divisor)], f"{number} = {number//divisor} × {divisor} + {ans}; remainder = {ans}.", "Dividend = Divisor × Quotient + Remainder")
            if mode == "odd_even":
                n=random.randint(10,100); ans="Even" if n%2==0 else "Odd"
                return mcq(f"Is {n} an odd or even number?", ans, ["Prime", "Composite", "Neither"], f"{n} {'is divisible by 2' if n%2==0 else 'is not divisible by 2'}, so it is {ans.lower()}.", "An even integer is divisible by 2")
            if mode == "divisors":
                n=random.choice([12,18,20,24,30,36,40,48]); ans=sum(1 for x in range(1,n+1) if n%x==0)
                return mcq(f"How many positive factors does {n} have?", str(ans), [str(ans+1),str(max(1,ans-1)),str(ans+2)], f"The positive factors of {n} number {ans} in total.", "Count the positive integers that divide the number exactly")
            n=random.randint(5,30); k=random.randint(2,12); ans=n*k
            return mcq(f"What is the smallest positive multiple of {n} that is greater than {ans-n}?", str(ans), [str(ans+n),str(ans-k),str(n+k)], f"Multiples of {n} are spaced by {n}; the required multiple is {n} × {k} = {ans}.", "Multiple = Integer × Number")
        if topic == "HCF & LCM":
            a=random.randint(6,30); b=random.randint(4,24)
            if random.choice([True,False]):
                ans=math.gcd(a,b); label="HCF"
                explanation=f"The greatest common divisor of {a} and {b} is {ans}."
            else:
                ans=math.lcm(a,b); label="LCM"
                explanation=f"The least common multiple of {a} and {b} is {ans}."
            return mcq(f"Find the {label} of {a} and {b}.", str(ans), [str(ans+a),str(max(1,ans-b)),str(a*b)], explanation, "HCF is the greatest common factor; LCM is the smallest common multiple")
        x=random.randint(2,15); y=random.randint(2,15); z=random.randint(2,12); ans=x+y*z
        return mcq(f"Evaluate {x} + {y} × {z}.", str(ans), [str((x+y)*z),str(x*y+z),str(ans+z)], f"Multiply first: {y} × {z} = {y*z}; then add {x} to get {ans}.", "Order of operations: brackets, powers, multiplication/division, addition/subtraction")

    if topic == "Simple & Compound Interest":
        p=random.choice([1000,1500,2000,2500,3000,4000,5000]); r=random.choice([5,8,10,12]); t=random.choice([1,2,3])
        mode=_next_reasoning_mode("Aptitude:" + topic + ":" + level, (["si","ci","difference"] if difficulty > 1 else ["si","ci"]))
        si=p*r*t/100; ci=p*((1+r/100)**t-1)
        if mode == "si": ans=si; q=f"Find the simple interest on ₹{p} at {r}% per annum for {t} year(s)."; exp=f"SI = PRT/100 = {p}×{r}×{t}/100 = ₹{ans:g}."; formula="Simple Interest = Principal × Rate × Time ÷ 100"
        elif mode == "ci": ans=ci; q=f"Find the compound interest on ₹{p} at {r}% per annum, compounded annually for {t} year(s)."; exp=f"Amount = {p}×(1+{r}/100)^{t} = ₹{p+ans:g}; CI = ₹{ans:g}."; formula="Compound Amount = P(1 + R/100)^T; CI = Amount − Principal"
        else: ans=ci-si; q=f"Find the difference between compound interest and simple interest on ₹{p} at {r}% for {t} years, compounded annually."; exp=f"SI = ₹{si:g}; CI = ₹{ci:g}; difference = ₹{ans:g}."; formula="Difference = Compound Interest − Simple Interest"
        ans=round(ans,2); return mcq(q, f"₹{ans:g}", [f"₹{ans+10:g}",f"₹{max(0,ans-10):g}",f"₹{ans*2:g}"], exp, formula)

    if topic in {"Time & Work", "Pipes & Cisterns"}:
        a=random.randint(4,18); b=random.randint(4,20); l=math.lcm(a,b); days_a=l//a; days_b=l//b
        if topic == "Pipes & Cisterns":
            fill=random.randint(4,15); drain=random.randint(8,25); net=1/fill-1/drain
            if net <= 0: drain=fill+random.randint(2,10); net=1/fill-1/drain
            answer=1/net
            return mcq(f"A pipe fills a tank in {fill} hours and a drain empties it in {drain} hours. If both are open, how long will the tank take to fill?", f"{answer:g} hours", [f"{fill+drain:g} hours",f"{(fill*drain)/(fill+drain):g} hours",f"{max(1,answer-2):g} hours"], f"Net rate = 1/{fill} − 1/{drain} = {net:g} tank/hour; time = {answer:g} hours.", "Combined rate = Filling rate − Emptying rate")
        mode=_next_reasoning_mode("Aptitude:" + topic + ":" + level, ["alone","together","workers"])
        if mode == "alone":
            return mcq(f"A can complete a job in {a} days. What fraction of the job does A complete in one day?", f"1/{a}", [f"1/{a+1}",f"{a}/1",f"1/{max(1,a-1)}"], f"One-day work = 1 ÷ {a} = 1/{a} of the job.", "One-day work = 1 ÷ Total days")
        together=1/(1/a+1/b)
        return mcq(f"A can finish a job in {a} days and B in {b} days. How many days will they take working together?", f"{together:g} days", [f"{a+b} days",f"{abs(a-b)} days",f"{(a+b)/2:g} days"], f"Combined rate = 1/{a} + 1/{b}; time = 1/(1/{a}+1/{b}) = {together:g} days.", "Combined work rate = Rate A + Rate B")

    if topic == "Mixtures & Alligation":
        a=random.randint(10,60); b=random.randint(a+5,90); target=random.randint(a+1,b-1); ratio_a=b-target; ratio_b=target-a
        g=math.gcd(ratio_a,ratio_b); ans=f"{ratio_a//g}:{ratio_b//g}"
        return mcq(f"Two solutions contain {a}% and {b}% acid. In what ratio should they be mixed to obtain {target}% acid?", ans, [f"{ratio_b//g}:{ratio_a//g}",f"{a}:{b}",f"{target}:{b-target}"], f"By alligation, ratio = ({b}−{target}):({target}−{a}) = {ans}.", "Alligation ratio = (Higher − Mean) : (Mean − Lower)")

    if topic == "Algebra & Equations":
        x=random.randint(2,20); a=random.randint(2,8); b=random.randint(1,20); c=a*x+b
        return mcq(f"Solve for x: {a}x + {b} = {c}.", str(x), [str(x+1),str(max(0,x-1)),str(c//a)], f"{a}x = {c} − {b} = {c-b}; x = {c-b}/{a} = {x}.", "Isolate the variable using inverse operations")

    if topic == "Progressions & Series":
        mode=_next_reasoning_mode("Aptitude:" + topic + ":" + level, ["ap","gp","difference"])
        if mode=="ap":
            first=random.randint(2,15); d=random.randint(2,8); n=random.randint(5,12); ans=first+(n-1)*d
            return mcq(f"Find the {n}th term of the AP {first}, {first+d}, {first+2*d}, ...", str(ans), [str(ans+d),str(ans-d),str(first+n*d)], f"aₙ = a + (n−1)d = {first}+({n}−1)×{d} = {ans}.", "AP nth term = a + (n − 1)d")
        if mode=="gp":
            first=random.choice([1,2,3,4]); ratio=random.choice([2,3]); n=random.randint(3,6); ans=first*ratio**(n-1)
            return mcq(f"Find the {n}th term of the GP starting {first}, {first*ratio}, {first*ratio**2}, ...", str(ans), [str(ans*ratio),str(ans//ratio),str(first*n)], f"aₙ = a×r^(n−1) = {first}×{ratio}^{n-1} = {ans}.", "GP nth term = a × r^(n − 1)")
        a=random.randint(3,30); d=random.randint(2,9); return mcq(f"Find the next number: {a}, {a+d}, {a+2*d}, {a+3*d}, ?", str(a+4*d), [str(a+3*d),str(a+5*d),str(a+4*d+1)], f"Each term increases by {d}; next = {a+3*d}+{d} = {a+4*d}.", "Identify the common difference or pattern")

    if topic == "Geometry & Mensuration":
        mode=_next_reasoning_mode("Aptitude:" + topic + ":" + level, ["rectangle","circle","triangle","cube"])
        if mode=="rectangle":
            l=random.randint(5,30); w=random.randint(3,20); ans=l*w
            return mcq(f"Find the area of a rectangle with length {l} cm and width {w} cm.", f"{ans} cm²", [f"{2*(l+w)} cm²",f"{l+w} cm²",f"{ans+ l} cm²"], f"Area = length × width = {l}×{w} = {ans} cm².", "Rectangle area = length × width")
        if mode=="circle":
            r=random.choice([7,14,21]); ans=22/7*r*r
            return mcq(f"Find the area of a circle with radius {r} cm (use π = 22/7).", f"{ans:g} cm²", [f"{2*22/7*r:g} cm²",f"{ans/2:g} cm²",f"{ans+22:g} cm²"], f"Area = πr² = 22/7×{r}×{r} = {ans:g} cm².", "Circle area = πr²")
        if mode=="triangle":
            base=random.randint(6,24); h=random.randint(4,20); ans=base*h/2
            return mcq(f"Find the area of a triangle with base {base} cm and height {h} cm.", f"{ans:g} cm²", [f"{base*h:g} cm²",f"{base+h:g} cm²",f"{ans+base:g} cm²"], f"Area = ½×base×height = ½×{base}×{h} = {ans:g} cm².", "Triangle area = ½ × base × height")
        side=random.randint(2,12); ans=side**3
        return mcq(f"Find the volume of a cube with side {side} cm.", f"{ans} cm³", [f"{6*side*side} cm³",f"{side*side} cm³",f"{ans+side} cm³"], f"Volume = side³ = {side}³ = {ans} cm³.", "Cube volume = side³")

    if topic == "Permutation & Combination":
        n=random.randint(5,9); r=random.randint(2,min(4,n)); mode=_next_reasoning_mode("Aptitude:" + topic + ":" + level, ["permutation","combination"]); ans=math.factorial(n)//math.factorial(n-r) if mode=="permutation" else math.factorial(n)//(math.factorial(r)*math.factorial(n-r))
        return mcq(f"In how many ways can {r} people be selected/arranged from {n} people? Use {'permutation (order matters)' if mode=='permutation' else 'combination (order does not matter)' }.", str(ans), [str(ans+r),str(max(1,ans-r)),str(math.factorial(n)//math.factorial(r))], f"{'nPr' if mode=='permutation' else 'nCr'} = {ans}.", "Permutation: n!/(n−r)!; Combination: n!/[r!(n−r)!]")

    if topic == "Probability":
        mode=_next_reasoning_mode("Aptitude:" + topic + ":" + level, ["coin","die","marble"])
        if mode=="coin":
            sides=random.choice([2,4,6,8]); return mcq(f"A fair coin is tossed {sides} times. What is the probability of getting heads on one specified toss?", "1/2", ["1/4","1/3","2/3"], "Each fair coin toss has two equally likely outcomes; probability of heads = 1/2.", "Probability = Favorable outcomes ÷ Total equally likely outcomes")
        if mode=="die":
            target=random.randint(1,6); return mcq(f"A fair six-sided die is rolled. What is the probability of rolling a {target}?", "1/6", ["1/2","1/3","1/5"], f"There is one favorable face among six, so probability = 1/6.", "Probability = Favorable outcomes ÷ Total outcomes")
        red=random.randint(2,8); blue=random.randint(2,8); return mcq(f"A bag contains {red} red and {blue} blue balls. What is the probability of drawing a red ball?", f"{red}/{red+blue}", [f"{blue}/{red+blue}",f"1/{red+blue}",f"{red}/{blue}"], f"Probability = red balls / total balls = {red}/{red+blue}.", "Probability = Favorable outcomes ÷ Total outcomes")

    if topic == "Set Theory & Venn Diagrams":
        a=random.randint(20,60); b=random.randint(20,60); both=random.randint(5,min(a,b)-1); ans=a+b-both
        return mcq(f"In a class, {a} students like tea, {b} like coffee, and {both} like both. How many like at least one?", str(ans), [str(a+b),str(ans+both),str(max(a,b))], f"n(A∪B)=n(A)+n(B)−n(A∩B)={a}+{b}−{both}={ans}.", "n(A ∪ B) = n(A) + n(B) − n(A ∩ B)")

    if topic == "Logarithms & Exponents":
        base=random.choice([2,3,5,10]); power=random.randint(2,5); value=base**power
        if _next_reasoning_mode("Aptitude:" + topic + ":" + level, ["exponent","logarithm"]) == "exponent":
            return mcq(f"Evaluate {base}^{power}.", str(value), [str(value+base),str(value-base),str(base*power)], f"{base}^{power} means multiplying {base} by itself {power} times = {value}.", "a^n = a multiplied by itself n times")
        return mcq(f"Evaluate log base {base} of {value}.", str(power), [str(power+1),str(max(0,power-1)),str(value)], f"Since {base}^{power} = {value}, log base {base} of {value} = {power}.", "log_b(a) = x means b^x = a")

    if topic == "Problems on Ages":
        age=random.randint(12,40); years=random.randint(3,12); future=age+years
        return mcq(f"A person is {age} years old now. How old will the person be after {years} years?", f"{future} years", [f"{age-years} years",f"{age+2*years} years",f"{future+1} years"], f"Future age = present age + years = {age}+{years} = {future}.", "Future age = Present age + elapsed years")

    if topic == "Clocks & Calendars":
        if random.choice([True,False]):
            h=random.randint(1,11); m=random.choice([0,10,15,20,30,40,45,50]); angle=abs(30*h-5.5*m); angle=min(angle,360-angle)
            return mcq(f"Find the smaller angle between the clock hands at {h}:{m:02d}.", f"{angle:g}°", [f"{abs(30*h+5.5*m):g}°",f"{(angle+30)%360:g}°",f"{max(0,angle-15):g}°"], f"Angle = |30H − 5.5M| = |30×{h} − 5.5×{m}| = {angle:g}° (smaller angle).", "Clock angle = |30H − 5.5M| degrees")
        days=random.randint(20,100); return mcq(f"How many odd days are there in {days} days?", str(days%7), [str((days+1)%7),str((days+2)%7),str(7-days%7)], f"Divide {days} by 7; remainder = {days%7} odd days.", "Odd days = Remainder after dividing total days by 7")

    if topic == "Data Interpretation":
        sales=[random.randint(20,100)*10 for _ in range(4)]; idx=random.randrange(4); names=["A","B","C","D"]; total=sum(sales)
        if _next_reasoning_mode("Aptitude:" + topic + ":" + level, ["read_value","percentage_share"]) == "read_value":
            ans=sales[idx]; return mcq(f"A table shows sales for products A-D as {dict(zip(names,sales))}. What are sales for product {names[idx]}?", str(ans), [str(ans+10),str(max(0,ans-10)),str(total)], f"Read product {names[idx]} from the given data: {ans}.", "Read the required value directly from the table")
        ans=round(sales[idx]/total*100,1)
        return mcq(f"Sales are {dict(zip(names,sales))}. What percentage of total sales came from product {names[idx]}?", f"{ans:g}%", [f"{ans+5:g}%",f"{max(0,ans-5):g}%",f"{100-ans:g}%"], f"Total = {total}; percentage = {sales[idx]}/{total}×100 = {ans:g}%.", "Percentage share = (Part ÷ Total) × 100")

    if topic == "Commercial Mathematics":
        cp=random.choice([500,800,1000,1200,1500]); rate=random.choice([5,10,15,20]); years=random.choice([1,2,3]); ans=cp*rate*years/100
        return mcq(f"A sum of ₹{cp} is invested at {rate}% simple interest for {years} year(s). What is the interest earned?", f"₹{ans:g}", [f"₹{ans+rate:g}",f"₹{max(0,ans-rate):g}",f"₹{cp+ans:g}"], f"Interest = {cp}×{rate}×{years}/100 = ₹{ans:g}.", "Simple interest = Principal × Rate × Time ÷ 100")

    return None
_REASONING_MODE_COUNTER = {}

def _next_reasoning_mode(topic, modes):
    """Cycle through methods so a round does not keep picking one pattern randomly."""
    key = str(topic)
    idx = _REASONING_MODE_COUNTER.get(key, 0)
    _REASONING_MODE_COUNTER[key] = idx + 1
    return modes[idx % len(modes)]

def make_local_reasoning_question(topic, level):
    """Generate topic-specific offline reasoning MCQs with rotating methods."""
    import math
    difficulty = {"Easy": 1, "Medium": 2, "Hard": 3}.get(level, 1)

    def mcq(q, answer, distractors, explanation, rule="Use the relationship or rule given in the question."):
        answer = str(answer)
        options = [answer]
        for item in distractors:
            item = str(item)
            if item not in options:
                options.append(item)
        while len(options) < 4:
            options.append(f"Option {len(options)+1}")
        options = options[:4]
        random.shuffle(options)
        return {"q": q, "options": options, "answer": answer,
                "explanation": explanation, "formula": rule}

    if topic == "Number Series":
        mode = _next_reasoning_mode(topic, ["add", "multiply", "squares", "alternate"] if difficulty > 1 else ["add", "multiply"])
        if mode == "add":
            start=random.randint(2,25); step=random.randint(2,9); seq=[start+i*step for i in range(5)]; ans=seq[-1]+step
            return mcq(f"Find the next number: {', '.join(map(str,seq))}, ?", ans, [ans+step,ans-1,ans+2*step], f"The common difference is {step}; next number = {seq[-1]} + {step} = {ans}.", "Check the difference between consecutive terms")
        if mode == "multiply":
            start=random.randint(1,5); ratio=random.randint(2,4); seq=[start*ratio**i for i in range(5)]; ans=seq[-1]*ratio
            return mcq(f"Find the next number: {', '.join(map(str,seq))}, ?", ans, [ans+ratio,ans//ratio,seq[-1]+ratio], f"Each term is multiplied by {ratio}; next = {seq[-1]} × {ratio} = {ans}.", "Check the multiplication ratio")
        if mode == "squares":
            start=random.randint(1,8); seq=[i*i for i in range(start,start+5)]; ans=(start+5)**2
            return mcq(f"Find the next number: {', '.join(map(str,seq))}, ?", ans, [ans-1,ans+2,(start+6)**2], f"These are consecutive squares; next = {start+5}² = {ans}.", "Recognize square-number patterns")
        a=random.randint(2,12); b=random.randint(2,12); seq=[a,b,a+1,b+2,a+2,b+4]; ans=a+3
        return mcq(f"Find the next number: {', '.join(map(str,seq))}, ?", ans, [b+6,ans+1,ans+2], f"Odd-position terms are {a}, {a+1}, {a+2}; the next odd-position term is {ans}.", "Separate odd-position and even-position terms")

    if topic == "Alphabet Series":
        mode = _next_reasoning_mode("Reasoning:" + topic + ":" + level, ["forward", "backward", "alternating"])
        start=random.randint(2,12); step=random.randint(1,4 if difficulty>1 else 3)
        if mode == "backward":
            indices=[(start-i*step)%26 for i in range(4)]; nxt=(indices[-1]-step)%26; rule=f"Move backward {step} letters each time."
        elif mode == "alternating":
            a,b=step,step+1; indices=[start, start+a, start+a-b, start+a-b+a]; nxt=(indices[-1]-b)%26; indices=[i%26 for i in indices]; rule=f"Alternate jumps of +{a}, −{b}, +{a}, then −{b}."
        else:
            indices=[(start+i*step)%26 for i in range(4)]; nxt=(indices[-1]+step)%26; rule=f"Move forward {step} letters each time."
        letters=[chr(65+i) for i in indices]; ans=chr(65+nxt)
        return mcq(f"Find the next letter: {', '.join(letters)}, ?", ans, [chr(65+(nxt+1)%26),chr(65+(nxt-1)%26),chr(65+(nxt+2)%26)], f"{rule} The next letter is {ans}.", "Convert letters to positions A=1 through Z=26")

    if topic in {"Coding-Decoding", "Coding-Decoding (Advanced)"}:
        # Rotate through genuinely different coding rules within each generated round.
        mode = _next_reasoning_mode(topic, [
            "uniform_shift", "reverse_alphabet", "reverse_word", "alphabet_positions",
            "word_value", "alternating_shift", "rearrangement"
        ])
        word = random.choice(["CAT", "DOG", "PEN", "MAP", "SUN", "BOX"])
        target = random.choice(["BAT", "CUP", "HEN", "TOP", "RED", "LIP"])
        alpha = lambda text: ''.join(chr((ord(ch.upper()) - 65) % 26 + 65) for ch in text)
        shift_word = lambda text, shifts: ''.join(chr((ord(ch) - 65 + shifts[i % len(shifts)]) % 26 + 65) for i, ch in enumerate(alpha(text)))

        if mode == "uniform_shift":
            shift = random.choice([1, 2, 3, -1] if difficulty > 1 else [1, 2])
            coded = shift_word(word, [shift]); result = shift_word(target, [shift])
            distractors = [shift_word(target, [shift + 1]), shift_word(target, [shift - 1]), target[::-1]]
            q = f"If {word} is coded as {coded}, how is {target} coded using the same letter shift?"
            exp = f"Each letter shifts by {shift} position(s); {target} becomes {result}."
            rule = "Apply the same forward/backward shift to every letter."
        elif mode == "reverse_alphabet":
            encode = lambda text: ''.join(chr(90 - (ord(ch) - 65)) for ch in alpha(text))
            coded, result = encode(word), encode(target)
            distractors = [target[::-1], shift_word(target, [1]), shift_word(target, [-1])]
            q = f"Using the reverse-alphabet code (A↔Z, B↔Y, C↔X, ...), {word} is coded as {coded}. How is {target} coded?"
            exp = f"Replace each letter with its opposite in the alphabet: {target} becomes {result}."
            rule = "Reverse alphabet mapping: letter position n maps to 27 − n."
        elif mode == "reverse_word":
            coded, result = word[::-1], target[::-1]
            distractors = [alpha(target), shift_word(target, [1]), target[1:] + target[:1]]
            q = f"A code reverses the order of letters: {word} becomes {coded}. How is {target} coded?"
            exp = f"Reverse the letter order: {target} becomes {result}."
            rule = "Write the letters in reverse order."
        elif mode == "alphabet_positions":
            encode = lambda text: '-'.join(str(ord(ch)-64) for ch in alpha(text))
            coded, result = encode(word), encode(target)
            distractors = ['-'.join(str((ord(ch)-64)%26+1) for ch in target[::-1]), encode(target[:2]), encode(word)]
            q = f"Letters are replaced by their alphabet positions (A=1, ..., Z=26). If {word} is coded as {coded}, how is {target} coded?"
            exp = f"{', '.join(target)} has positions {result}."
            rule = "Convert each letter to its position: A=1 through Z=26."
        elif mode == "word_value":
            value = lambda text: sum(ord(ch)-64 for ch in alpha(text))
            coded, result = str(value(word)), str(value(target))
            distractors = [str(value(target)+1), str(value(target)-1), str(value(target)+3)]
            q = f"A word is coded as the sum of its letter positions. If {word} is coded as {coded}, what is the code for {target}?"
            exp = f"Add the letter positions of {target}: {' + '.join(str(ord(ch)-64) for ch in target)} = {result}."
            rule = "Word value = sum of alphabet positions of all letters."
        elif mode == "alternating_shift":
            shifts = [1, -1] if difficulty == 1 else [2, -1]
            coded, result = shift_word(word, shifts), shift_word(target, shifts)
            distractors = [shift_word(target, [shifts[0], shifts[0]]), shift_word(target, [-shifts[0], shifts[1]]), target[::-1]]
            q = f"Code letters by alternating shifts {shifts[0]:+d}, {shifts[1]:+d}, {shifts[0]:+d}, ... . If {word} becomes {coded}, how is {target} coded?"
            exp = f"Apply the alternating shifts to {target}: the result is {result}."
            rule = "Apply the first shift to positions 1, 3, 5... and the second to positions 2, 4, 6..."
        else:
            encode = lambda text: alpha(text)[1::2] + alpha(text)[::2]
            coded, result = encode(word), encode(target)
            distractors = [alpha(target)[::-1], alpha(target)[::2] + alpha(target)[1::2], shift_word(target, [1])]
            q = f"A code moves letters in odd positions after letters in even positions. If {word} becomes {coded}, how is {target} coded?"
            exp = f"For {target}, even-position letters come first, followed by odd-position letters: {result}."
            rule = "Rearrange letters as positions 2,4,6... followed by positions 1,3,5..."

        return mcq(q, result, distractors, exp, rule)

    if topic == "Analogy":
        # Several relationship families with large pools so a round does not run out
        # after the original six fixed questions.
        mode = _next_reasoning_mode("Reasoning:Analogy:" + level, [
            "animal_home", "young_adult", "object_function", "person_place",
            "part_whole", "opposites", "synonyms", "number_rule"
        ])
        pools = {
            "animal_home": [
                ("Bird", "Nest", "Bee", "Hive"), ("Lion", "Den", "Rabbit", "Burrow"),
                ("Horse", "Stable", "Pig", "Sty"), ("Spider", "Web", "Ant", "Anthill"),
                ("Dog", "Kennel", "Chicken", "Coop"), ("Fox", "Earth", "Squirrel", "Drey"),
                ("Cow", "Shed", "Bee", "Hive"), ("Owl", "Hollow", "Bear", "Cave"),
            ],
            "young_adult": [
                ("Puppy", "Dog", "Kitten", "Cat"), ("Calf", "Cow", "Foal", "Horse"),
                ("Lamb", "Sheep", "Cub", "Lion"), ("Kid", "Goat", "Piglet", "Pig"),
                ("Duckling", "Duck", "Chick", "Chicken"), ("Fawn", "Deer", "Cygnet", "Swan"),
            ],
            "object_function": [
                ("Pen", "Write", "Knife", "Cut"), ("Key", "Unlock", "Eraser", "Erase"),
                ("Broom", "Sweep", "Shovel", "Dig"), ("Thermometer", "Measure temperature", "Scale", "Measure weight"),
                ("Clock", "Show time", "Compass", "Show direction"), ("Scissors", "Cut", "Needle", "Sew"),
                ("Camera", "Photograph", "Microscope", "Magnify"), ("Oven", "Bake", "Refrigerator", "Cool"),
            ],
            "person_place": [
                ("Doctor", "Hospital", "Teacher", "School"), ("Chef", "Kitchen", "Pilot", "Cockpit"),
                ("Librarian", "Library", "Judge", "Court"), ("Farmer", "Farm", "Scientist", "Laboratory"),
                ("Waiter", "Restaurant", "Cashier", "Shop"), ("Actor", "Stage", "Referee", "Field"),
            ],
            "part_whole": [
                ("Finger", "Hand", "Toe", "Foot"), ("Page", "Book", "Room", "House"),
                ("Wheel", "Car", "Petal", "Flower"), ("Branch", "Tree", "Key", "Keyboard"),
                ("Chapter", "Novel", "Player", "Team"), ("Letter", "Word", "Brick", "Wall"),
            ],
            "opposites": [
                ("Day", "Night", "Hot", "Cold"), ("Up", "Down", "In", "Out"),
                ("Victory", "Defeat", "Success", "Failure"), ("Ancient", "Modern", "Early", "Late"),
                ("Expand", "Contract", "Accept", "Reject"), ("Generous", "Stingy", "Brave", "Cowardly"),
            ],
            "synonyms": [
                ("Happy", "Joyful", "Angry", "Furious"), ("Quick", "Rapid", "Silent", "Quiet"),
                ("Begin", "Start", "End", "Finish"), ("Tiny", "Small", "Huge", "Enormous"),
                ("Assist", "Help", "Purchase", "Buy"), ("Honest", "Truthful", "Clever", "Smart"),
            ],
        }
        if mode == "number_rule":
            a = random.randint(2, 20)
            multiplier = random.choice([2, 3, 4, 5])
            offset = random.choice([1, 2, 3, 5, 7])
            c = random.randint(2, 20)
            b = a * multiplier + offset
            ans_num = c * multiplier + offset
            distractors = [str(c * multiplier - offset), str(c + multiplier + offset), str(c * multiplier)]
            return mcq(f"Complete the analogy: {a} is to {b} as {c} is to __.", str(ans_num), distractors,
                       f"The rule is ×{multiplier} then +{offset}: {c} × {multiplier} + {offset} = {ans_num}.",
                       f"Apply the same arithmetic rule to both numbers: ×{multiplier}, then +{offset}")
        a, b, c, ans = random.choice(pools[mode])
        distractors = list(dict.fromkeys([a, b, c]))
        while len(distractors) < 3:
            distractors.append(random.choice([x[3] for group in pools.values() if group for x in group]))
        return mcq(f"Complete the analogy: {a} is to {b} as {c} is to __.", ans, distractors[:3],
                   f"The relationship is {a} → {b}; applying the same relationship gives {c} → {ans}.",
                   f"Identify the relationship type: {mode.replace('_', ' ')}")

    if topic == "Blood Relations":
        if difficulty == 1:
            gender=random.choice(["son","daughter"]); rel="mother"
            return mcq(f"Ravi is the {gender} of Meena. Meena is Ravi's __.", rel, ["sister","aunt","cousin"], f"If Ravi is Meena's {gender}, Meena is Ravi's mother.", "Translate the statement one relationship at a time")
        scenarios=[("A is B's sister. B is C's father. How is A related to C?","Aunt",["Mother","Sister","Grandmother"]),
                   ("P is Q's son. Q is R's daughter. How is P related to R?","Grandson",["Nephew","Brother","Father"]),
                   ("M is N's brother. N is O's mother. How is M related to O?","Uncle",["Father","Cousin","Grandfather"])]
        q,ans,opts=random.choice(scenarios)
        return mcq(q,ans,opts,f"Follow each family link in order; the correct relationship is {ans.lower()}.","Draw a small family tree to track generations")

    if topic == "Direction Sense":
        directions=[("north","south"),("south","north"),("east","west"),("west","east")]
        start=random.choice(["north","south","east","west"]); turns=random.randint(1,3); steps=[random.randint(2,8) for _ in range(2 if difficulty==1 else 3)]
        turn=random.choice(["left","right"]); order=["north","east","south","west"]; idx=order.index(start); idx=(idx+(1 if turn=="right" else -1)*turns)%4; final=order[idx]
        return mcq(f"A person faces {start.upper()} and turns {turn} {turns} time(s), 90° each time. Which direction are they facing?", final.title(), [d.title() for d in ["north","south","east","west"] if d!=final][:3], f"Each {turn} turn changes direction by 90°. After {turns} turn(s), the person faces {final}.", "Track turns clockwise for right and anticlockwise for left")

    if topic == "Ranking & Order":
        total=random.randint(20,60); left=random.randint(3,total//2); right=total-left+1; ans=total-left+1
        return mcq(f"A student is {left}th from the left and {right}th from the right. How many students are in the row?", ans, [total,total+1,total-1], f"Total = left rank + right rank − 1 = {left} + {right} − 1 = {ans}.", "Total people = rank from one side + rank from the other side − 1")

    if topic == "Seating Arrangement":
        people=["A","B","C","D","E","F"]
        if difficulty == 1:
            return mcq("Four people A, B, C and D sit in a row. A is left of B, and C is right of B. Which person is definitely to the left of C?", "B", ["D","C","Cannot be determined"], "Because A is left of B and C is right of B, B must be left of C.", "Convert each clue into a left/right relation")
        order=people[:]; random.shuffle(order); i=random.randint(0,3); x,y=order[i],order[i+1]
        return mcq(f"In a row facing north, the order from left to right is {', '.join(order)}. Who sits immediately to the right of {x}?", order[i+1], [order[(i+2)%6],order[(i-1)%6],order[(i+3)%6]], f"Read the given left-to-right order; {y} is immediately to the right of {x}.", "For people facing north, their right is the viewer's right")

    if topic == "Syllogisms":
        cases=[("All pens are stationery. All stationery are useful. Which conclusion follows?","All pens are useful",["All useful things are pens","No pen is useful","Some pens are not stationery"]),
               ("No cats are dogs. Some pets are cats. Which conclusion follows?","Some pets are not dogs",["All pets are dogs","Some dogs are cats","No pets are cats"]),
               ("All engineers are graduates. Some graduates are artists. Must some engineers be artists?","No, it does not necessarily follow",["Yes, all engineers are artists","Yes, some artists are engineers","All graduates are engineers"])]
        q,ans,opts=random.choice(cases)
        return mcq(q,ans,opts,f"Use only the statements given. The supported conclusion is: {ans}.","Do not assume a relationship that is not stated")

    if topic == "Statement & Conclusion":
        cases=[("Statement: The library will remain closed on Sunday. Conclusion: A student cannot borrow a book from the library on Sunday.","Conclusion follows",["Conclusion does not follow","Statement is unrelated","Both are contradictory"]),
               ("Statement: All applicants must submit an ID. Conclusion: An applicant without an ID has not met the stated requirement.","Conclusion follows",["Conclusion does not follow","All applicants are rejected","ID is optional"]),
               ("Statement: Some buses are electric. Conclusion: All buses are electric.","Conclusion does not follow",["Conclusion follows","No bus is electric","All electric vehicles are buses"])]
        q,ans,opts=random.choice(cases)
        return mcq(q+" Choose the correct evaluation.",ans,opts,f"Check whether the conclusion is guaranteed by the statement alone: {ans.lower()}.","A conclusion must be supported by the given statement")

    if topic == "Mathematical Operations":
        a,b,c=[random.randint(2,12) for _ in range(3)]; mode=_next_reasoning_mode("Reasoning:" + topic + ":" + level, ["swap_plus_minus","swap_times_divide"])
        if mode=="swap_plus_minus":
            q=f"If '+' means '−' and '−' means '+', evaluate {a} + {b} − {c}."; ans=a-b+c; exp=f"Replace symbols: {a} + {b} − {c} becomes {a} − {b} + {c} = {ans}."
        else:
            q=f"If '×' means '+' and '+' means '×', evaluate {a} × {b} + {c}."; ans=a+b*c; exp=f"Replace symbols: {a} × {b} + {c} becomes {a} + {b} × {c} = {ans} using order of operations."
        return mcq(q,ans,[ans+2,ans-2,a*b+c],exp,"Replace each operation sign before calculating")

    if topic == "Data Sufficiency":
        n=random.randint(8,30); d=random.randint(2,8)
        q=f"What is the value of x? (1) x + {d} = {n}. (2) x is a positive integer. Which information is sufficient?"
        ans="Statement (1) alone is sufficient"
        return mcq(q,ans,["Statement (2) alone is sufficient","Both statements together are required","Neither statement is sufficient"],f"Statement (1) gives x = {n} − {d} = {n-d}, a unique value. Statement (2) alone does not.","A statement is sufficient if it determines exactly one answer")

    if topic == "Calendar & Clock Reasoning":
        if _next_reasoning_mode("Reasoning:" + topic + ":" + level, ["day_shift","clock_addition"]) == "day_shift":
            days=random.randint(10,100); ans=days%7; day=random.choice(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]); days_names=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]; target=days_names[(days_names.index(day)+ans)%7]
            return mcq(f"If today is {day}, what day will it be {days} days from today?",target,[days_names[(days_names.index(target)+1)%7],days_names[(days_names.index(target)-1)%7],day],f"{days} ÷ 7 leaves remainder {ans}; move forward {ans} day(s) from {day} to get {target}.","Day shifts repeat every 7 days")
        h=random.randint(1,11); ans= (h+6)%12 or 12
        return mcq(f"What hour will it be 6 hours after {h} o'clock?",f"{ans} o'clock",[f"{(ans+1-1)%12 or 12} o'clock",f"{(ans+2-1)%12 or 12} o'clock",f"{h} o'clock"],f"Add 6 hours to {h} and wrap around a 12-hour clock: {ans}.","Clock arithmetic wraps around after 12")

    if topic == "Logical Reasoning":
        mode=_next_reasoning_mode(topic, ["ordering", "true_statement", "comparison", "deduction"])
        if mode == "ordering":
            names=random.sample(["Asha","Bala","Charan","Divya"], 4)
            vals=sorted(random.sample(range(150,195),4), reverse=True)
            who=random.choice(names)
            mapping=dict(zip(names, vals))
            tallest=max(mapping, key=mapping.get)
            return mcq(f"Heights are: {', '.join(f'{n}={mapping[n]} cm' for n in names)}. Who is tallest?", tallest, [n for n in names if n!=tallest][:3], f"The greatest height is {mapping[tallest]} cm, belonging to {tallest}.", "Compare the values stated in the question")
        if mode == "true_statement":
            a=random.randint(4,15); b=random.randint(2, a-1); c=a+b
            return mcq(f"If x={a} and y={b}, which statement is true?", f"x + y = {c}", [f"x − y = {c}", f"x × y = {c}", f"x + y = {c+1}"], f"Substitute the values: {a}+{b}={c}.", "Check each option against the given facts")
        if mode == "comparison":
            a,b,c=random.sample(range(10,70),3)
            biggest=max(a,b,c); smallest=min(a,b,c)
            return mcq(f"P={a}, Q={b}, R={c}. Which is the smallest value?", {a:"P",b:"Q",c:"R"}[smallest], [n for n,v in [("P",a),("Q",b),("R",c)] if v!=smallest][:3], f"Compare {a}, {b}, and {c}; the smallest is {smallest}.", "Translate the conditions into a simple comparison")
        return mcq("A notice says: 'Employees must wear their ID cards inside the office.' Ravi is inside the office without his ID card. Which conclusion is supported?", "Ravi is not following the notice", ["Ravi is a visitor", "Ravi has no job", "The notice is optional"], "The notice requires employees inside the office to wear ID cards; the described behaviour does not follow it.", "Use only the information explicitly provided")

    if topic == "Odd One Out":
        mode=_next_reasoning_mode(topic, ["squares", "primes", "fruit", "letters"])
        if mode == "squares":
            vals=random.sample([4,9,16,25,36,49,64,81,100,121],3); odd=random.choice([12,18,20,30,45,50])
            items=vals+[odd]; random.shuffle(items); ans=str(odd)
            return mcq(f"Choose the odd one out: {', '.join(map(str,items))}.", ans, ["All are perfect squares", "None", "Cannot be determined"], f"All the other values are perfect squares; {odd} is not.", "Find the property shared by three options")
        if mode == "primes":
            vals=random.sample([2,3,5,7,11,13,17,19,23,29],3); odd=random.choice([9,15,21,25,27,33]); items=vals+[odd]; random.shuffle(items)
            return mcq(f"Choose the odd one out: {', '.join(map(str,items))}.", odd, ["All are prime", str(max(items)+1), str(min(items)-1)], f"The other three numbers are prime; {odd} is composite.", "Check whether each number has factors other than 1 and itself")
        if mode == "fruit":
            items=["Apple","Mango","Carrot","Banana"]; random.shuffle(items)
            return mcq(f"Choose the odd one out: {', '.join(items)}.", "Carrot", ["Apple","Mango","Banana"], "Carrot is commonly classified as a vegetable; the others are fruits.", "Classify the words by category")
        items=["ACEG","BDFH","CEGI","DFIK"]; odd=random.choice(items)
        # Select a known irregular letter pattern as the odd item.
        odd="DFIK"; items=["ACEG","BDFH","CEGI",odd]; random.shuffle(items)
        return mcq(f"Choose the odd letter group: {', '.join(items)}.", odd, [x for x in items if x!=odd][:3], "The first three groups advance by two letters consistently; DFIK has a different final jump.", "Compare the letter positions within each group")

    return None


def build_round(category, topic, level, count, client=None, seen=None):
    """Build a fresh round, asking AI for new questions and filtering repeats."""
    seen_questions = list(seen or [])
    used = {normalize_question(q) for q in seen_questions}
    selected = []
    last_ai_error = None

    # Ask AI in more than one batch if it returns duplicates or too few questions.
    if client is not None:
        for attempt in range(2):
            needed = count - len(selected)
            if needed <= 0:
                break
            excluded = seen_questions + [item["q"] for item in selected]
            prompt = f"""
You are an expert placement aptitude question setter.
Generate exactly {needed} original multiple-choice questions for placement practice.
Category: {category}
Topic: {topic}
Difficulty: {level}
Round attempt: {attempt + 1}. Generate a DIFFERENT set from previous attempts.

Return ONLY a valid JSON array. Every object must contain:
q (clear question text), options (exactly four distinct strings),
answer (exactly one option string), explanation (step-by-step solution),
formula (formula/rule or 'No formula required').

Rules:
- Match the requested topic and difficulty exactly.
- Use varied values, scenarios, and solution methods.
- Do not paraphrase or repeat any excluded question.
- Ensure the correct answer is actually present in options and is mathematically correct.
- For aptitude questions, include a worked calculation in explanation.
- Avoid trick wording and ambiguous answers.
Excluded questions (do not repeat): {json.dumps(excluded[-250:], ensure_ascii=False)}
"""
            raw = ask_gemini(client, prompt)
            if raw and raw.startswith("Gemini request failed:"):
                last_ai_error = raw
                break
            parsed = parse_json_array(raw)
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                q_text = str(item.get("q", "")).strip()
                options = item.get("options")
                if not isinstance(options, list):
                    continue
                clean_options = [str(option).strip() for option in options]
                answer = str(item.get("answer", "")).strip()
                norm = normalize_question(q_text)
                if (not q_text or norm in used or len(clean_options) != 4
                        or len({o.casefold() for o in clean_options}) != 4
                        or answer not in clean_options):
                    continue
                item["q"] = q_text
                item["options"] = clean_options
                item["answer"] = answer
                item["formula"] = str(item.get("formula") or "No formula required")
                item["explanation"] = str(
                    item.get("explanation") or
                    "Apply the relevant formula or rule and work through the values step by step."
                )
                selected.append(item)
                used.add(norm)
                if len(selected) >= count:
                    return selected[:count]

    # Generate fresh local questions when Gemini is unavailable or busy.
    if category in {"Aptitude", "Reasoning"}:
        attempts = 0
        while len(selected) < count and attempts < max(300, count * 60):
            attempts += 1
            if category == "Aptitude":
                question = make_local_aptitude_question(topic, level)
            else:
                question = make_local_reasoning_question(topic, level)
            # Some topics/levels do not have a local generator. Do not stop
            # the entire round on the first unsupported pattern; continue the
            # bounded loop and then use the question-bank fallback below.
            if not question:
                continue
            norm = normalize_question(question.get("q", ""))
            if norm and norm not in used:
                selected.append(question)
                used.add(norm)

    # Final fallback for topics without a local generator.
    bank_questions = list(QUESTION_BANK.get(category, {}).get(topic, {}).get(level, []))
    random.shuffle(bank_questions)
    for question in bank_questions:
        if len(selected) >= count:
            break
        norm = normalize_question(question.get("q", ""))
        if norm and norm not in used:
            selected.append(question)
            used.add(norm)

    if last_ai_error:
        st.warning("Gemini is temporarily unavailable; local questions were used where possible. Details: " + last_ai_error)
    return selected[:count]


# ----------------------------- APP PASSWORD / SIDEBAR -----------------
def password_hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000).hex()


def get_saved_password():
    with get_connection() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS app_auth (id INTEGER PRIMARY KEY CHECK (id = 1), salt TEXT NOT NULL, password_hash TEXT NOT NULL)"
        )
        return conn.execute("SELECT salt, password_hash FROM app_auth WHERE id = 1").fetchone()


def save_app_password(password):
    salt = secrets.token_bytes(16)
    hashed = password_hash(password, salt)
    with get_connection() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS app_auth (id INTEGER PRIMARY KEY CHECK (id = 1), salt TEXT NOT NULL, password_hash TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO app_auth (id, salt, password_hash) VALUES (1, ?, ?)",
            (salt.hex(), hashed),
        )


# Gemini key is no longer entered in the sidebar. Configure it privately in
# the GEMINI_API_KEY environment variable or Streamlit secrets instead.
try:
    api_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    api_key = ""
api_key = api_key or os.environ.get("GEMINI_API_KEY", "")

st.sidebar.title("🎯 Placement Prep")
student = st.sidebar.text_input("Student name", value="Student").strip() or "Student"
st.sidebar.caption("Local progress is stored in placement.db on this computer.")

saved_password = get_saved_password()
if not st.session_state.get("app_authenticated", False):
    st.title("🔐 Placement Prep Login")
    if saved_password is None:
        st.write("Create a password to protect this placement preparation app on this computer.")
        with st.form("set_app_password_form"):
            new_password = st.text_input("Create password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            setup_submit = st.form_submit_button("Set password")
        if setup_submit:
            if len(new_password) < 8:
                st.error("Use at least 8 characters for your password.")
            elif new_password != confirm_password:
                st.error("The passwords do not match.")
            else:
                save_app_password(new_password)
                st.session_state["app_authenticated"] = True
                st.success("Password created. You are logged in.")
                st.rerun()
    else:
        st.write("Enter your app password to continue.")
        with st.form("app_login_form"):
            entered_password = st.text_input("Password", type="password")
            login_submit = st.form_submit_button("Log in")
        if login_submit:
            salt_bytes = bytes.fromhex(saved_password[0])
            candidate_hash = password_hash(entered_password, salt_bytes)
            if hmac.compare_digest(candidate_hash, saved_password[1]):
                st.session_state["app_authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect password. Try again.")
    st.stop()

if st.sidebar.button("Log out"):
    st.session_state["app_authenticated"] = False
    st.rerun()

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Aptitude Practice",
        "Reasoning Practice",
        "Coding Practice",
        "AI Mock Interview",
        "Progress History",
    ],
)
client = get_client(api_key) if api_key else None


# ----------------------------- DASHBOARD ------------------------------
if page == "Dashboard":
    st.title("🎯 AI Placement Preparation & Tracking System")
    st.write("Practice, review your weak areas, and track your placement preparation in one place.")

    df = load_results(student)
    if df.empty:
        st.info("No practice results yet. Start with Aptitude Practice or Reasoning Practice.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Practice sessions", "0")
        col2.metric("Average score", "—")
        col3.metric("Latest score", "—")
    else:
        df["percent"] = df.apply(lambda r: score_percent(r["score"], r["total"]), axis=1)
        col1, col2, col3 = st.columns(3)
        col1.metric("Practice sessions", len(df))
        col2.metric("Average score", f"{df['percent'].mean():.1f}%")
        col3.metric("Latest score", f"{df.iloc[0]['percent']:.1f}%")

        st.subheader("Recent activity")
        st.dataframe(
            df[["created_at", "category", "topic", "score", "total", "difficulty"]].head(10),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Performance by category")
        category_df = df.groupby("category", as_index=False)["percent"].mean()
        st.bar_chart(category_df.set_index("category")["percent"])

        st.subheader("Areas to improve")
        topic_df = df.groupby("topic", as_index=False)["percent"].mean().sort_values("percent")
        if not topic_df.empty:
            st.dataframe(topic_df.head(5), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("How adaptive practice works")
    st.markdown(
        "- **80% or higher:** move to the next difficulty level.\n"
        "- **Below 50%:** move down one level when possible and revise basics.\n"
        "- **50–79%:** stay at the same level.\n"
        "- Difficulty levels: Easy → Medium → Hard."
    )


# ----------------------------- APTITUDE / REASONING -------------------
elif page in ["Aptitude Practice", "Reasoning Practice"]:
    category = "Aptitude" if page == "Aptitude Practice" else "Reasoning"
    st.title("📝 " + page)
    topic = st.selectbox("Choose topic", question_topics(category), key=f"topic_select_{category}")
    st.caption(f"Active concept: {topic} — questions are generated for this concept only.")
    start_level = st.selectbox("Starting difficulty", LEVELS, index=0)
    count = st.select_slider("Questions in this round", options=[5, 10, 15, 20, 30, 50, 60], value=10)

    prefix = f"quiz_{category.lower()}_"
    state_key = prefix + "data"
    seen_key = f"seen_questions_{category}_{topic}"
    # SQLite is the permanent source of history; session state is a quick cache.
    if seen_key not in st.session_state:
        st.session_state[seen_key] = load_seen_questions(student, category, topic)
    else:
        persisted = load_seen_questions(student, category, topic)
        st.session_state[seen_key] = list(dict.fromkeys(persisted + st.session_state[seen_key]))

    if st.button("Start / Restart round", type="primary"):
        seen_questions = st.session_state[seen_key]
        questions = build_round(category, topic, start_level, count, client, seen_questions)
        if not questions:
            st.warning(
                "No new questions are available in the built-in bank for this topic. "
                "Enter your Gemini API key in the sidebar to generate fresh questions, "
                "or choose another topic."
            )
        else:
            st.session_state[seen_key] = list(dict.fromkeys(seen_questions + [q["q"] for q in questions]))
            save_seen_questions(student, category, topic, [q["q"] for q in questions])
            st.session_state[state_key] = {
                "questions": questions,
                "level": start_level,
                "topic": topic,
                "category": category,
                "answers": {},
                "submitted": False,
                "round_id": datetime.now().strftime("%Y%m%d%H%M%S"),
            }

    quiz = st.session_state.get(state_key)
    if quiz and quiz["topic"] != topic:
        st.info("Your current round is for another topic. Click Start / Restart round to load this topic.")
    elif quiz:
        st.caption(f"Difficulty: **{quiz['level']}** · Topic: **{quiz['topic']}**")
        if len(quiz["questions"]) < count:
            st.warning(
                f"Only {len(quiz['questions'])} new question(s) could be created for this round. "
                "Try a different topic or level."
            )
        with st.form(prefix + "answer_form"):
            for i, q in enumerate(quiz["questions"]):
                st.markdown(f"**Q{i + 1}. {q['q']}**")
                chosen = st.radio(
                    f"Choose answer for question {i + 1}",
                    q["options"],
                    index=None,
                    key=f"{prefix}{quiz['round_id']}_q{i}",
                    label_visibility="collapsed",
                )
                quiz["answers"][str(i)] = chosen
                st.divider()
            submitted = st.form_submit_button("Submit answers")

        if submitted and not quiz["submitted"]:
            score = sum(
                1 for i, q in enumerate(quiz["questions"])
                if quiz["answers"].get(str(i)) == q["answer"]
            )
            total = len(quiz["questions"])
            quiz["submitted"] = True
            quiz["score"] = score
            quiz["total"] = total
            quiz["next_level"] = difficulty_after_round(score, total, quiz["level"])
            save_result(
                student, category, topic, score, total, quiz["level"],
                notes=f"Adaptive next level: {quiz['next_level']}",
            )
            st.rerun()

        if quiz.get("submitted"):
            score = quiz["score"]
            total = quiz["total"]
            st.success(f"Your score: {score}/{total} ({score_percent(score, total)}%)")
            for i, q in enumerate(quiz["questions"]):
                chosen = quiz["answers"].get(str(i))
                correct = chosen == q["answer"]
                st.markdown(f"**Q{i + 1}. {q['q']}**")
                st.write(f"Your answer: {chosen or 'Not answered'}")
                st.write(f"Correct answer: **{q['answer']}**")
                if not correct:
                    st.error("Incorrect answer — review the explanation below.")
                else:
                    st.success("Correct answer!")
                formula = q.get("formula")
                if not formula:
                    formula_by_topic = {
                        "Percentages": "Percentage = (Part / Whole) × 100; Percentage value = (Percentage / 100) × Total",
                        "Profit & Loss": "Profit = Selling Price − Cost Price; Loss = Cost Price − Selling Price; Profit/Loss % = (Profit or Loss / Cost Price) × 100",
                        "Time, Speed & Distance": "Speed = Distance / Time; Distance = Speed × Time; Time = Distance / Speed",
                        "Ratio & Proportion": "Share = (Individual ratio part / Sum of ratio parts) × Total amount",
                        "Averages": "Average = Sum of values / Number of values",
                        "Number Series": "Look for a pattern in differences, multiplication, division, or squares",
                        "Logical Reasoning": "Use only the facts given; compare each statement against those facts",
                        "Odd One Out": "Identify the common property shared by most options",
                        "Coding-Decoding": "Apply the same letter/number transformation consistently",
                    }
                    formula = formula_by_topic.get(quiz["topic"], "No formula required")
                if formula and formula != "No formula required":
                    st.markdown(f"**Formula / rule:** `{formula}`")
                st.markdown(f"**Step-by-step explanation:** {q['explanation']}")
                st.divider()

            if quiz["next_level"] != quiz["level"]:
                st.info(f"Based on your score, your next recommended level is **{quiz['next_level']}**.")
            else:
                st.info(f"Your next recommended level is **{quiz['level']}**. Keep practicing!")

            if st.button("Continue with next adaptive round", type="primary", key=prefix + "continue"):
                next_level = quiz["next_level"]
                seen_questions = st.session_state.get(seen_key, [])
                next_questions = build_round(category, topic, next_level, count, client, seen_questions)
                if not next_questions:
                    st.warning(
                        "The built-in bank has no unused questions for this topic. "
                        "Enter your Gemini API key to generate fresh questions, or choose another topic."
                    )
                else:
                    st.session_state[seen_key] = list(dict.fromkeys(seen_questions + [q["q"] for q in next_questions]))
                    save_seen_questions(student, category, topic, [q["q"] for q in next_questions])
                    st.session_state[state_key] = {
                        "questions": next_questions,
                        "level": next_level,
                        "topic": topic,
                        "category": category,
                        "answers": {},
                        "submitted": False,
                        "round_id": datetime.now().strftime("%Y%m%d%H%M%S"),
                    }
                    st.rerun()
    else:
        st.info("Choose a topic and click **Start / Restart round** to begin.")


# ----------------------------- CODING PRACTICE ------------------------
elif page == "Coding Practice":
    st.title("💻 Coding & Technical Practice")
    st.caption("Practice programming, SQL, DBMS, DSA, Operating Systems and Computer Networks.")

    SUBJECT_TOPICS = {
        "Python": ["Variables and Input", "Conditions", "Loops", "Strings", "Lists and Arrays", "Functions", "Dictionaries and Sets", "Recursion", "OOP", "Searching and Sorting", "List Comprehension", "File Handling", "Exception Handling", "Lambda and Map Filter", "Two Pointers", "Sliding Window", "Stack and Queue", "Decorators", "Generators", "Date and Time"],
        "Java": ["Basics and Input", "Conditions and Loops", "Arrays", "Strings", "Methods", "OOP", "Inheritance and Polymorphism", "Collections", "Exception Handling", "Searching and Sorting", "ArrayList and HashMap", "Interfaces", "File Handling", "Multithreading", "Recursion", "Generics", "Lambda Expressions", "Binary Search", "Stack and Queue", "Pattern Printing"],
        "C": ["Basics and Input", "Conditions and Loops", "Arrays", "Strings", "Functions", "Pointers", "Structures", "Dynamic Memory", "Recursion", "Searching and Sorting", "Pointer Arithmetic", "File Handling", "Bit Manipulation", "Linked Lists", "Stacks and Queues", "Matrix Operations", "String Functions", "Command Line Arguments", "Unions and Enums", "Preprocessor"],
        "C++": ["Basics and Input", "Conditions and Loops", "Arrays and Strings", "Functions", "Pointers and References", "OOP", "STL", "Recursion", "Searching and Sorting", "Data Structures", "Vectors and Pairs", "Maps and Sets", "Templates", "Operator Overloading", "File Handling", "Bit Manipulation", "Linked Lists", "Trees", "Graphs", "Dynamic Programming"],
        "SQL": ["SELECT and WHERE", "ORDER BY and DISTINCT", "Aggregate Functions", "GROUP BY and HAVING", "INNER JOIN", "LEFT and RIGHT JOIN", "Subqueries", "CASE Expressions", "Constraints", "Window Functions", "Date Functions", "String Functions", "CTE", "UNION and UNION ALL", "INSERT UPDATE DELETE", "Self Join", "Indexes and Query Tuning", "Ranking Functions", "Duplicate Detection", "Conditional Aggregation"],
        "DBMS": ["Keys", "Normalization", "ER Model", "SQL vs NoSQL", "Transactions and ACID", "Concurrency Control", "Indexing", "Views", "Deadlocks", "Database Architecture"],
        "Data Structures & Algorithms": ["Arrays", "Strings", "Linked Lists", "Stacks and Queues", "Hashing", "Trees", "Graphs", "Binary Search", "Sorting", "Dynamic Programming"],
        "Operating Systems": ["Processes and Threads", "CPU Scheduling", "Process Synchronization", "Deadlocks", "Memory Management", "Paging and Segmentation", "Virtual Memory", "File Systems", "System Calls", "Scheduling Algorithms"],
        "Computer Networks": ["OSI Model", "TCP/IP Model", "IP Addressing", "Subnetting", "TCP vs UDP", "DNS and DHCP", "HTTP and HTTPS", "Routing", "Network Devices", "Security Basics"],
        "OOP Concepts": ["Classes and Objects", "Encapsulation", "Abstraction", "Inheritance", "Polymorphism", "Overloading vs Overriding", "Constructors", "Interfaces", "Access Modifiers", "SOLID Basics"],
    }
    PROGRAM_TASKS = {
        "Variables and Input": ("Swap two numbers", "Read two numbers and print them after swapping. Example: input 4 9 → output 9 4.", "Store one value temporarily, then exchange the two values."),
        "Conditions": ("Largest of three numbers", "Read three integers and print the largest without sorting them.", "Compare the first two, then compare their larger value with the third."),
        "Loops": ("Sum from 1 to N", "Read a positive integer N and print 1 + 2 + ... + N.", "Start a running total at zero and add each number from 1 through N."),
        "Strings": ("Palindrome check", "Read a string and determine whether it reads the same forward and backward.", "Compare the string with its reversed form; decide how to handle letter case and spaces."),
        "Lists and Arrays": ("Second largest distinct value", "Given a list of integers, print the second-largest distinct value. Handle lists without such a value.", "Track the largest and second-largest distinct values in one pass."),
        "Arrays": ("Second largest distinct value", "Given an array, find the second-largest distinct value without sorting.", "Keep track of the largest and runner-up while scanning once."),
        "Functions": ("Factorial using a function", "Write a function that returns N! for a non-negative integer N.", "Use a loop from 2 through N; remember that 0! and 1! equal 1."),
        "Recursion": ("Recursive sum", "Write a recursive function to calculate the sum of integers from 1 to N.", "Define a base case for N <= 0 and reduce N on each recursive call."),
        "Searching and Sorting": ("Binary search", "Given a sorted array and a target, return its index or -1 if absent.", "Maintain left and right boundaries and halve the search interval."),
        "SELECT and WHERE": ("Filter records", "Table Employee(id, name, department, salary). Write a query to show names of employees in the 'IT' department earning more than 50000.", "Use SELECT with a WHERE clause combining two conditions."),
        "Aggregate Functions": ("Average salary", "For Employee(id, name, department, salary), return the average salary.", "Use AVG() on the salary column."),
        "GROUP BY and HAVING": ("Department salary totals", "Show each department and its total salary, only for departments whose total exceeds 200000.", "Group by department, then filter groups with HAVING."),
        "INNER JOIN": ("Employee department names", "Employee(id, name, dept_id) and Department(dept_id, dept_name). Return each employee name with the department name.", "Join on the shared dept_id column."),
        "Subqueries": ("Above-average salary", "Return employees whose salary is greater than the average salary of all employees.", "Compare salary against a scalar subquery using AVG()."),
    }
    THEORY_QUESTIONS = {
        "DBMS": {
            "Easy": [("Which key uniquely identifies each row in a table?", ["Foreign key", "Primary key", "Candidate relationship", "Duplicate key"], 1, "A primary key uniquely identifies each row and cannot be NULL."), ("What does SQL stand for?", ["Structured Query Language", "Sequential Question Language", "System Query Link", "Simple Queue Logic"], 0, "SQL stands for Structured Query Language."), ("Which ACID property means a transaction is all-or-nothing?", ["Consistency", "Isolation", "Atomicity", "Durability"], 2, "Atomicity ensures all operations in a transaction happen or none do.")],
            "Medium": [("Which normal form primarily removes partial dependency on a composite key?", ["1NF", "2NF", "3NF", "BCNF only"], 1, "Second Normal Form removes partial dependencies of non-key attributes on part of a candidate key."), ("Which command usually makes a transaction's changes permanent?", ["ROLLBACK", "COMMIT", "SAVEPOINT", "GRANT"], 1, "COMMIT finalizes the transaction's changes."), ("What is a database index mainly used for?", ["Encrypt every row", "Speed up data retrieval", "Replace primary keys", "Prevent all duplicates"], 1, "Indexes can speed up lookups, though they add storage and write overhead.")],
            "Hard": [("Which isolation anomaly occurs when a transaction reads data written by an uncommitted transaction?", ["Dirty read", "Lost update", "Phantom read only", "Checkpoint"], 0, "A dirty read observes uncommitted data that may later be rolled back."), ("A relation is in BCNF when every determinant is a…", ["Foreign key", "Superkey", "Nullable column", "Non-prime attribute"], 1, "In BCNF, every non-trivial functional dependency has a superkey as its determinant."), ("Which technique helps recover a database after a crash using recorded changes?", ["Normalization", "Logging", "Projection", "Cartesian product"], 1, "Write-ahead logging records changes so recovery can redo or undo operations.")],
        },
        "Operating Systems": {
            "Easy": [("Which is an example of an operating system?", ["Python", "Linux", "HTML", "SQL"], 1, "Linux is an operating system."), ("A process is…", ["A program in execution", "Only a source file", "A network cable", "A database table"], 0, "A process is a running instance of a program."), ("Which memory is volatile?", ["ROM", "RAM", "SSD", "DVD"], 1, "RAM generally loses its contents when power is removed.")],
            "Medium": [("Which scheduling algorithm uses a fixed time slice?", ["FCFS", "Round Robin", "SJF only", "FIFO page replacement"], 1, "Round Robin assigns each ready process a time quantum."), ("Which condition means resources cannot be forcibly taken from a process?", ["Mutual exclusion", "No preemption", "Circular wait", "Hold and wait"], 1, "No preemption means resources are released voluntarily rather than forcibly taken."), ("What does a page fault indicate?", ["A requested page is not currently mapped in RAM", "CPU is broken", "Disk is full", "A process has terminated"], 0, "A page fault occurs when a process accesses a page that needs OS handling, often because it is not resident.")],
            "Hard": [("Thrashing is most closely associated with…", ["Excessive paging", "High network bandwidth", "Disk formatting", "Dead code"], 0, "Thrashing occurs when the system spends excessive time paging instead of executing."), ("A semaphore is primarily used for…", ["Synchronization", "Compilation", "Compression", "IP assignment"], 0, "Semaphores coordinate access to shared resources and synchronize processes or threads."), ("Which is necessary for deadlock?", ["Circular wait", "High clock speed", "Large RAM", "Multithreading is disabled"], 0, "Circular wait is one of the four Coffman conditions required for deadlock.")],
        },
        "Computer Networks": {
            "Easy": [("Which protocol is commonly used to load web pages securely?", ["HTTPS", "FTP only", "ARP", "SMTP"], 0, "HTTPS is HTTP protected by TLS."), ("Which device forwards packets between networks?", ["Router", "Keyboard", "Repeater only", "Monitor"], 0, "Routers forward IP packets between networks."), ("How many layers are in the OSI model?", ["4", "5", "7", "8"], 2, "The OSI reference model has seven layers." )],
            "Medium": [("Which transport protocol is connection-oriented?", ["UDP", "TCP", "IP", "ARP"], 1, "TCP establishes a connection and provides reliable ordered delivery."), ("DNS primarily translates…", ["Domain names to IP addresses", "MAC addresses to passwords", "HTML to SQL", "Files to packets"], 0, "DNS resolves domain names to records such as IP addresses."), ("Which address is used for delivery within a local Ethernet network?", ["MAC address", "URL path", "Port name", "Process ID"], 0, "Ethernet frames use MAC addresses for local-link delivery.")],
            "Hard": [("Which IPv4 prefix corresponds to a subnet mask of 255.255.255.192?", ["/24", "/25", "/26", "/27"], 2, "The mask has 26 one-bits, so the prefix is /26."), ("Which protocol resolves an IPv4 address to a MAC address on a local network?", ["DNS", "ARP", "DHCP", "TLS"], 1, "ARP maps an IPv4 address to a link-layer address on a local network."), ("What does TCP's three-way handshake establish?", ["A TCP connection", "A DNS zone", "A Wi-Fi password", "A subnet mask"], 0, "SYN, SYN-ACK and ACK establish the TCP connection and synchronize sequence numbers.")],
        }
    }

    subject = st.selectbox("Choose subject", list(SUBJECT_TOPICS.keys()), key="coding_subject")
    level = st.select_slider("Difficulty level", options=["Easy", "Medium", "Hard"], value="Easy", key="coding_level")

    topic = st.selectbox("Choose topic", SUBJECT_TOPICS[subject], key="coding_topic")

    # Keep the selected subject, topic, and level fixed. Only increment the
    # question number when Next Question is clicked.
    def _coding_question_state_key():
        return f"coding_question_index_{subject}_{topic}_{level}"

    question_index_key = _coding_question_state_key()
    if question_index_key not in st.session_state:
        st.session_state[question_index_key] = 0

    if subject in ["DBMS", "Operating Systems", "Computer Networks"]:
        qset = THEORY_QUESTIONS[subject][level]
        qidx_key = f"tech_qidx_{subject}_{level}"
        if qidx_key not in st.session_state:
            st.session_state[qidx_key] = 0
        idx = st.session_state[qidx_key] % len(qset)
        q, options, answer_idx, explanation = qset[idx]
        st.subheader(f"{subject} · {level} · Question {idx + 1}")
        st.write(q)
        chosen = st.radio("Select your answer", options, key=f"tech_answer_{subject}_{level}_{idx}")
        if st.button("Check answer", key=f"tech_check_{subject}_{level}_{idx}"):
            if options.index(chosen) == answer_idx:
                st.success("Correct! 🎉")
                save_result(student, subject, topic, 1, 1, level, notes="Technical MCQ correct")
            else:
                st.error("Not quite. Try reviewing the concept.")
                st.info("Hint: " + explanation.split(".")[0] + ".")
            st.caption("Explanation: " + explanation)
        if st.button("Next question", key=f"tech_next_{subject}_{level}"):
            st.session_state[qidx_key] = (idx + 1) % len(qset)
            st.rerun()
    else:
        # Each difficulty has its own task bank. Do not reuse the Easy task at
        # Medium/Hard; this was the reason every level displayed the same program.
        LEVEL_TASKS = {
            "Variables and Input": {
                "Easy": ("Swap two numbers", "Read two numbers and print them after swapping. Example: input 4 9 -> output 9 4.", "Use a temporary variable to preserve one value while swapping."),
                "Medium": ("Rotate three values", "Read integers a, b, c and rotate them so the old b becomes a, old c becomes b, and old a becomes c. Print the new values.", "Save the original values before overwriting them."),
                "Hard": ("Normalize a time duration", "Read a non-negative number of seconds and print the equivalent hours, minutes, and remaining seconds. Example: 3672 -> 1 1 12.", "Use integer division and remainder; avoid floating-point division.")},
            "Basics and Input": {
                "Easy": ("Read and print values", "Read two integers and print their sum.", "Convert input to integers before adding."),
                "Medium": ("Digit statistics", "Read a positive integer and print its digit count and digit sum.", "Repeatedly use remainder by 10 and integer division by 10."),
                "Hard": ("Number base conversion", "Read a non-negative decimal integer and convert it to binary without using a built-in conversion function.", "Repeatedly divide by 2 and collect remainders in reverse order.")},
            "Conditions": {
                "Easy": ("Largest of three numbers", "Read three integers and print the largest.", "Compare values without sorting."),
                "Medium": ("Triangle validity and type", "Read three side lengths. First determine whether they form a valid triangle; if valid, classify it as equilateral, isosceles, or scalene.", "Check the triangle inequality before classifying sides."),
                "Hard": ("Quadratic equation roots", "Read coefficients a, b, c. Handle the linear case when a=0 and otherwise report whether roots are real and calculate them when possible.", "Use the discriminant b*b - 4*a*c and handle a=0 separately.")},
            "Loops": {
                "Easy": ("Sum from 1 to N", "Read N and print the sum of integers from 1 through N.", "Keep a running total in a loop."),
                "Medium": ("Prime number test", "Read an integer N and print whether it is prime.", "Check divisors only up to the square root of N."),
                "Hard": ("Print all primes in a range", "Read L and R and print every prime number in the inclusive range [L, R]. Handle values below 2.", "Test each candidate efficiently, stopping divisors at its square root.")},
            "Strings": {
                "Easy": ("Palindrome check", "Read a string and determine whether it reads the same forward and backward.", "Compare the string with its reverse."),
                "Medium": ("First non-repeating character", "Given a string, print the first character that occurs exactly once; print -1 if none exists.", "Count character frequencies, then scan the original string in order."),
                "Hard": ("Longest substring without repeats", "Given a string, find the length of the longest substring containing no repeated characters.", "Use a sliding window and track the last position of each character.")},
            "Lists and Arrays": {
                "Easy": ("Find the maximum", "Read N integers and print the maximum value.", "Initialize the maximum from the first item."),
                "Medium": ("Second largest distinct value", "Find the second-largest distinct array value without sorting. Print -1 if it does not exist.", "Track the largest and second-largest distinct values in one pass."),
                "Hard": ("Maximum subarray sum", "Find the maximum sum of any non-empty contiguous subarray in an integer array.", "Use a running best-ending-here sum and a global best (Kadane's algorithm).")},
            "Arrays": {
                "Easy": ("Find the maximum", "Read an array and print its maximum value.", "Scan each element and update the maximum."),
                "Medium": ("Rotate array by K", "Rotate an array to the right by K positions; K may be larger than the array length.", "Reduce K modulo the array length and use slicing or reversal."),
                "Hard": ("Maximum subarray sum", "Find the maximum sum of a non-empty contiguous subarray in O(n) time.", "Use Kadane's algorithm and account for all-negative arrays.")},
            "Functions": {
                "Easy": ("Factorial function", "Write a function that returns N! for a non-negative integer N.", "Multiply integers from 2 through N; 0! is 1."),
                "Medium": ("GCD function", "Write a function that returns the greatest common divisor of two non-negative integers.", "Apply Euclid's algorithm repeatedly using remainder."),
                "Hard": ("Memoized Fibonacci", "Write a function to compute the Nth Fibonacci number efficiently for N up to 1000.", "Use memoization or iterative dynamic programming; consider large integer output.")},
            "Recursion": {
                "Easy": ("Recursive sum", "Use recursion to calculate 1 + 2 + ... + N.", "Define a base case and reduce N each call."),
                "Medium": ("Recursive power", "Implement power(base, exponent) recursively for a non-negative exponent in O(log exponent) time.", "Use exponentiation by squaring and halve the exponent."),
                "Hard": ("Generate balanced parentheses", "Given N pairs of parentheses, generate all valid balanced combinations.", "Backtrack while never using more closing than opening parentheses.")},
            "Searching and Sorting": {
                "Easy": ("Linear search", "Find the index of a target in an array, or -1 if it is absent.", "Scan the array from left to right."),
                "Medium": ("Binary search", "Given a sorted array and target, return its index or -1 in O(log n) time.", "Maintain left and right bounds and halve the search interval."),
                "Hard": ("Search in rotated sorted array", "Search for a target in a rotated sorted array of distinct integers in O(log n) time.", "At each step identify which half is sorted and decide where the target can lie.")},
            "SELECT and WHERE": {
                "Easy": ("Filter employees", "From Employee(id, name, department, salary), select names of IT employees earning over 50000.", "Use WHERE with both conditions."),
                "Medium": ("Filter and sort employees", "Return the top five employees in the IT department with salary above 50000, ordered by salary descending.", "Combine WHERE, ORDER BY DESC, and a row limit."),
                "Hard": ("Conditional filtering", "Return employees whose salary is above their department's average salary.", "Compare each employee with a grouped average, using a correlated subquery or join.")},
            "Aggregate Functions": {
                "Easy": ("Average salary", "Return the average salary from Employee(id, name, department, salary).", "Use AVG(salary)."),
                "Medium": ("Salary range by department", "For each department, return minimum, maximum, and average salary; sort by average salary descending.", "Group by department and use MIN, MAX, AVG."),
                "Hard": ("Departments above overall average", "List departments whose average salary is greater than the average salary across the whole company.", "Use GROUP BY and compare grouped averages with a scalar subquery.")},
            "GROUP BY and HAVING": {
                "Easy": ("Count employees by department", "Show each department and its employee count.", "Group by department and use COUNT."),
                "Medium": ("Department salary totals", "Show departments whose total salary exceeds 200000.", "Group by department and filter groups with HAVING."),
                "Hard": ("Top earning departments", "Return departments whose average salary is above the company average, ordered by average salary descending.", "Use GROUP BY, HAVING with a subquery, and ORDER BY.")},
            "INNER JOIN": {
                "Easy": ("Employee department names", "Join Employee(id, name, dept_id) with Department(dept_id, dept_name) and show each employee with the department name.", "Match the shared dept_id columns."),
                "Medium": ("Department employee counts", "Show each department name and the number of employees assigned to it.", "Join the tables and group by department; consider whether empty departments matter."),
                "Hard": ("Employees earning above department average", "Return employee names, department names, and salaries for employees earning above their own department's average.", "Join the tables and compare salary against a department-specific aggregate.")},
            "Subqueries": {
                "Easy": ("Above-average salary", "Return employees whose salary is greater than the average salary of all employees.", "Compare salary with a scalar AVG subquery."),
                "Medium": ("Second-highest salary", "Return the second-highest distinct salary from Employee; handle duplicate salaries.", "Use DISTINCT with ordering and a subquery, or a ranking function."),
                "Hard": ("Nth-highest salary", "Return the Nth-highest distinct salary for a supplied N, without returning duplicates as separate ranks.", "Rank distinct salary values with DENSE_RANK or use nested subqueries.")},
        }
        EXTRA_LEVEL_TASKS = {
            "List Comprehension": {
                "Easy": ("Filter even numbers", "Given a list of integers, create a new list containing only even values using a list comprehension.", "Use the condition n % 2 == 0 inside the comprehension."),
                "Medium": ("Flatten a matrix", "Convert a list of lists into one flat list using a comprehension.", "Use a nested comprehension: iterate over each row, then each value."),
                "Hard": ("Build frequency pairs", "From a list of words, create a dictionary mapping each distinct word to its frequency using comprehensions or related built-ins.", "Use set/dict comprehensions and count each unique word carefully.")},
            "File Handling": {
                "Easy": ("Count file lines", "Read a text file and print the number of lines, handling a missing file gracefully.", "Use a context manager and catch FileNotFoundError."),
                "Medium": ("Word frequency from file", "Read a text file and print the five most frequent words after converting to lowercase and removing basic punctuation.", "Normalize text, split into words, count with a dictionary or Counter."),
                "Hard": ("Merge sorted files", "Read two files containing sorted integers and write one merged sorted output file without loading both files entirely into memory.", "Use two iterators and merge like the merge step of merge sort.")},
            "Exception Handling": {
                "Easy": ("Safe integer input", "Convert a user-provided string to an integer and display a friendly message for invalid input.", "Wrap int conversion in try/except ValueError."),
                "Medium": ("Validated calculator", "Create a calculator that handles invalid numbers and division by zero without crashing.", "Handle conversion errors separately from ZeroDivisionError."),
                "Hard": ("Retrying input parser", "Build a parser that keeps accepting input until a valid integer in a specified range is entered, with a maximum of three failed attempts.", "Track attempts and validate both conversion and range.")},
            "Sliding Window": {
                "Easy": ("Maximum sum of K elements", "Given an integer array and K, find the maximum sum of any contiguous subarray of length K.", "Calculate the first window, then add the entering value and subtract the leaving value."),
                "Medium": ("Longest substring without repeats", "Find the length of the longest substring containing no repeated characters.", "Use a sliding window and remember the last position of each character."),
                "Hard": ("Minimum window substring", "Given strings s and t, return the shortest substring of s containing every character of t with multiplicity.", "Use a frequency map and expand/shrink a sliding window.")},
            "Two Pointers": {
                "Easy": ("Pair sum in sorted array", "Given a sorted array and target, determine whether two values sum to the target.", "Move the left or right pointer based on the current sum."),
                "Medium": ("Three sum", "Return unique triplets in an integer array that sum to zero.", "Sort, fix one value, then use two pointers while skipping duplicates."),
                "Hard": ("Trapping rain water", "Given bar heights, compute how much rain water can be trapped.", "Use two pointers and track the maximum height seen from each side.")},
            "Stack and Queue": {
                "Easy": ("Balanced brackets", "Check whether (), [], and {} are properly balanced in a string.", "Push opening brackets; each closing bracket must match the stack top."),
                "Medium": ("Next greater element", "For each array value, find the next greater value to its right, or -1 if none exists.", "Use a monotonic stack and process indices from right to left."),
                "Hard": ("Sliding window maximum", "For every window of size K, output its maximum in O(n) time.", "Use a deque of indices while removing expired and smaller values.")},
            "Bit Manipulation": {
                "Easy": ("Count set bits", "Count the number of 1 bits in a non-negative integer.", "Repeatedly clear the lowest set bit with n &= n - 1."),
                "Medium": ("Find single number", "Every array value appears twice except one. Find the unique value in O(n) time and O(1) extra space.", "XOR cancels equal pairs."),
                "Hard": ("Two unique numbers", "Every value appears twice except two distinct values. Find both unique values in linear time and constant extra space.", "XOR all values, isolate one set bit of the combined XOR, then partition values by that bit.")},
            "CTE": {
                "Easy": ("Employees above salary threshold", "Using a CTE, select employees earning more than 50000 from Employee(id, name, department, salary).", "Define a WITH clause, then query its named result."),
                "Medium": ("Department average salary", "Use a CTE to calculate each department's average salary and list departments above 60000 average.", "Create a grouped CTE and filter its result."),
                "Hard": ("Recursive number series", "Write a recursive CTE that generates integers from 1 through 100.", "Use an anchor row and a recursive SELECT with a stopping condition.")},
            "Window Functions": {
                "Easy": ("Rank employees", "Show each employee and their salary rank using RANK() ordered by salary descending.", "Use RANK() OVER (ORDER BY salary DESC)."),
                "Medium": ("Top three salaries per department", "Return the three highest-paid employees in each department, preserving ties.", "Use DENSE_RANK or RANK partitioned by department."),
                "Hard": ("Running total", "For Transactions(id, account_id, amount, txn_date), calculate each account's running total ordered by date.", "Use SUM(amount) OVER (PARTITION BY account_id ORDER BY txn_date).")},
            "Date Functions": {
                "Easy": ("Current date report", "Write a query to show employee names and their joining date, filtering records after a chosen date.", "Use a date literal appropriate to your SQL dialect."),
                "Medium": ("Monthly sales", "For Sales(sale_id, sale_date, amount), return total sales grouped by year and month.", "Extract year/month or group by a truncated date."),
                "Hard": ("Consecutive activity days", "For Activity(user_id, activity_date), find users with activity on at least three consecutive dates.", "Compare dates using self joins or date differences and deduplicate rows.")},
            "String Functions": {
                "Easy": ("Uppercase names", "Return employee names in uppercase and their character lengths.", "Use UPPER() and LENGTH()/LEN() as supported by the database."),
                "Medium": ("Find duplicate emails", "From Users(id, email), return normalized emails that occur more than once, ignoring case and surrounding spaces.", "Group by a normalized expression and use HAVING COUNT(*) > 1."),
                "Hard": ("Parse name components", "Given a table of full names, return first and last name columns, documenting your assumptions about spaces.", "Use string-position and substring functions for your SQL dialect.")},
            "UNION and UNION ALL": {
                "Easy": ("Combine contact lists", "Combine customer emails and supplier emails into one deduplicated list.", "UNION removes duplicates; ensure both SELECTs return compatible columns."),
                "Medium": ("Keep source labels", "Combine customer and supplier names with a source column identifying the origin, retaining duplicates.", "Use UNION ALL and a literal source label in each SELECT."),
                "Hard": ("Compare yearly records", "Combine current-year and archive transaction tables, then find IDs that appear in both sources.", "Use UNION ALL with source labels, then aggregate by transaction ID.")},
        }
        if topic in EXTRA_LEVEL_TASKS:
            task_title, task_description, task_hint = EXTRA_LEVEL_TASKS[topic][level]
        elif topic in LEVEL_TASKS:
            task_title, task_description, task_hint = LEVEL_TASKS[topic][level]
        elif topic in PROGRAM_TASKS and level == "Easy":
            task_title, task_description, task_hint = PROGRAM_TASKS[topic]
        else:
            # Topic-specific prompt with increasing requirements when no dedicated bank entry exists.
            if level == "Easy":
                task_title = f"Fundamentals: {topic}"
                task_description = f"Solve one basic {subject} exercise on {topic}. Implement the core operation and handle ordinary input."
                task_hint = f"Identify the key concept in {topic} and work through a small example by hand."
            elif level == "Medium":
                task_title = f"Applied challenge: {topic}"
                task_description = f"Solve an applied {subject} problem on {topic}. Handle boundary cases and explain the time/space complexity or query logic. Do not just implement the basic example."
                task_hint = f"Break {topic} into steps, consider at least three edge cases, and choose an efficient approach."
            else:
                task_title = f"Advanced challenge: {topic}"
                task_description = f"Solve an advanced {subject} problem on {topic}. Aim for an efficient solution, handle tricky edge cases, and state complexity or trade-offs."
                task_hint = f"Look for an optimized approach to {topic}; consider worst-case input, invariants, and edge cases before coding."
        question_number = st.session_state[question_index_key] + 1
        if question_number > 1:
            # Follow-up questions stay within the SAME selected topic and level.
            # These prompts are distinct variants, not navigation to another topic.
            variant_number = question_number - 1
            if level == "Easy":
                variant_templates = [
                    ("Practice {n}: Basic application", "Create a new {subject} solution focused on {topic}. Use a different example from the previous question, accept the required input, and print the expected result.", "Write down the input and output first, then apply the core {topic} operation step by step."),
                    ("Practice {n}: Boundary cases", "Solve another {subject} problem on {topic}. Make sure your solution handles empty, minimum, maximum, or repeated values wherever they apply.", "List the normal case and at least two boundary cases before implementing {topic}."),
                    ("Practice {n}: Small variation", "Implement a different beginner-level task using {topic} in {subject}. Keep the solution simple and handle invalid or unusual input where relevant.", "Break the task into input, processing, and output; test a tiny example manually."),
                ]
            elif level == "Medium":
                variant_templates = [
                    ("Practice {n}: Applied challenge", "Create a new medium-level {subject} problem about {topic}. Include multiple conditions or data items and handle edge cases; do not repeat the previous task.", "Identify the key operation, then test cases with duplicates, boundaries, and typical input."),
                    ("Practice {n}: Efficiency challenge", "Solve a different medium-level problem involving {topic} in {subject}. Prefer a clear and efficient approach, and explain its time/space complexity or query logic.", "Choose a suitable data structure or query operation and estimate its cost before coding."),
                    ("Practice {n}: Robust solution", "Build another medium-level solution for {topic} in {subject}, including validation and at least three meaningful test cases.", "Separate the main logic into steps and check what happens at boundary values."),
                ]
            else:
                variant_templates = [
                    ("Practice {n}: Advanced challenge", "Solve a new hard-level {subject} problem focused on {topic}. Aim for an optimized approach and handle tricky edge cases; this must differ from the previous question.", "State the invariant or core idea, then analyze worst-case complexity and edge cases."),
                    ("Practice {n}: Optimization", "Design an advanced {subject} solution using {topic}, prioritizing performance and correctness on large inputs. Explain trade-offs where relevant.", "Look for repeated work, choose an efficient structure or algorithm, and test worst-case input."),
                    ("Practice {n}: Edge-case challenge", "Create another advanced problem on {topic} in {subject} that combines the core concept with boundary cases or constraints. Do not reuse the prior task.", "Write adversarial test cases first, then build a solution that preserves the key invariant."),
                ]
            title_template, description_template, hint_template = variant_templates[(question_number - 2) % len(variant_templates)]
            task_title = title_template.format(n=question_number)
            task_description = description_template.format(subject=subject, topic=topic)
            task_hint = hint_template.format(topic=topic)

        # Give the first three Easy Python Variables-and-Input practices
        # concrete requirements so they can be checked locally even when Gemini
        # is unavailable. The checks below compare exact stdout for each case.
        if subject == "Python" and topic == "Variables and Input" and level == "Easy":
            if question_number == 1:
                task_title = "Swap two numbers"
                task_description = (
                    "Read two integers from separate input lines. Swap their values and "
                    "print the swapped numbers on one line, separated by one space. "
                    "Example input: 4 then 9. Expected output: 9 4."
                )
                task_hint = "Read both values before changing either one; test negative and equal values too."
            elif question_number == 2:
                task_title = "Calculate rectangle area"
                task_description = (
                    "Read the rectangle's length and width from separate input lines. "
                    "They may be decimal values. Print exactly `Area: ` followed by "
                    "the area (length multiplied by width). Example: inputs 5 and 3 "
                    "produce `Area: 15.0`. Assume dimensions are non-negative."
                )
                task_hint = "Convert both inputs to numbers, multiply them, and match the required output label and format."
            elif question_number == 3:
                task_title = "Sum, smallest, and largest"
                task_description = (
                    "Read exactly three integers (they may be on separate lines or one line). "
                    "Print three lines in this exact format: `Sum: <sum>`, `Smallest: <minimum>`, "
                    "and `Largest: <maximum>`. If fewer or more than three values are supplied, "
                    "print `Please enter exactly 3 integers.` If any of the three values is not "
                    "an integer, print `Invalid input. Please enter integers.`"
                )
                task_hint = "Split the received input into tokens, validate the count and integers, then calculate sum, minimum, and maximum. Test duplicates, negatives, and empty input."
            elif question_number == 4:
                task_title = "Celsius to Fahrenheit converter"
                task_description = (
                    "Read one Celsius temperature from standard input; it may be an integer or decimal. "
                    "Print exactly `Fahrenheit: ` followed by the converted Fahrenheit temperature rounded "
                    "to two decimal places. Use Fahrenheit = Celsius * 9 / 5 + 32. If input is empty or "
                    "not a valid number, print exactly `Invalid input`."
                )
                task_hint = "Read one value, validate that it can be converted to a number, apply the conversion formula, and format the result to two decimal places. Test zero, a negative value, decimals, invalid text, and empty input."

        st.subheader(f"{subject} · {topic} · {level} · Question {question_number}")
        st.markdown(f"**Problem: {task_title}**")
        st.write(task_description)
        st.markdown("**Sample input/output:** Use the requirements above to define suitable sample cases; test edge cases too.")
        with st.expander("Show hint (no full solution)"):
            st.write(task_hint)
        if st.button("➡️ Next Question", key=f"coding_next_{subject}_{topic}_{level}"):
            st.session_state[question_index_key] += 1
            st.rerun()
        if subject == "SQL":
            st.caption("Use this practice schema: Employee(id, name, department, salary); Department(dept_id, dept_name). Query execution can be added with an isolated SQLite runner.")
        code = st.text_area("Write your solution", height=230, key=f"solution_{subject}_{topic}_{level}", placeholder=f"Write your {subject} solution here...")
        if st.button("Review my solution", type="primary", key=f"review_{subject}_{topic}_{level}"):
            if not code.strip():
                st.warning("Write your solution first.")
            elif subject == "Python" and topic == "Variables and Input" and level == "Easy" and question_number in {1, 2, 3, 4}:
                # Deterministic local tests: do not depend on Gemini availability.
                if question_number == 1:
                    test_cases = [
                        ("4\n9\n", "9 4"),
                        ("-2\n7\n", "7 -2"),
                        ("5\n5\n", "5 5"),
                    ]
                elif question_number == 2:
                    test_cases = [
                        ("5\n3\n", "Area: 15.0"),
                        ("2.5\n4\n", "Area: 10.0"),
                        ("0\n8\n", "Area: 0.0"),
                    ]
                elif question_number == 3:
                    test_cases = [
                        ("10\n20\n30\n", "Sum: 60\nSmallest: 10\nLargest: 30"),
                        ("5\n5\n5\n", "Sum: 15\nSmallest: 5\nLargest: 5"),
                        ("-10\n0\n10\n", "Sum: 0\nSmallest: -10\nLargest: 10"),
                        ("", "Please enter exactly 3 integers."),
                        ("1\nx\n3\n", "Invalid input. Please enter integers."),
                    ]
                else:
                    test_cases = [
                        ("25\n", "Fahrenheit: 77.00"),
                        ("0\n", "Fahrenheit: 32.00"),
                        ("-40\n", "Fahrenheit: -40.00"),
                        ("36.5\n", "Fahrenheit: 97.70"),
                        ("hello\n", "Invalid input"),
                        ("", "Invalid input"),
                    ]
                with st.spinner("Running sample and edge test cases..."):
                    result = run_python_test_cases(code, test_cases)
                if result["status"] == "syntax_error":
                    st.error(result["message"])
                    st.info("Hint: Fix the syntax issue, then review again. The full solution is not shown.")
                else:
                    passed_count = sum(item["passed"] for item in result["results"])
                    if passed_count == len(test_cases):
                        st.success(f"All {passed_count} test cases passed!")
                        st.write("Your output matches the expected output for the available tests.")
                    else:
                        st.warning(f"{passed_count}/{len(test_cases)} test cases passed. Each test is run separately with the input shown below.")
                        for item in result["results"]:
                            label = "Passed" if item["passed"] else "Needs attention"
                            with st.expander(f"Test case {item['case']} — {label}", expanded=not item["passed"]):
                                st.markdown("**Input provided automatically by the test runner:**")
                                st.code(item.get("input", "").rstrip("\n") or "(empty input)", language="text")
                                st.markdown("**Expected output:**")
                                st.code(item.get("expected", "") or "(no output expected)", language="text")
                                st.markdown("**Your program output:**")
                                st.code(item.get("actual", "") or "(no standard output produced)", language="text")
                                if item.get("error_details"):
                                    st.markdown("**Error details / traceback:**")
                                    st.code(item["error_details"], language="text")
                        if question_number == 1:
                            local_hint = "Read both input values before swapping them, and print the two swapped values in the required order."
                        elif question_number == 2:
                            local_hint = "Convert both inputs to numeric values, multiply them, and match the exact `Area: ...` output format."
                        elif question_number == 3:
                            local_hint = "Validate that exactly three integers were supplied; also check repeated values, negative values, empty input, and the required output labels."
                        else:
                            local_hint = "Validate the single temperature input, apply Celsius * 9 / 5 + 32, format to exactly two decimal places, and handle invalid or empty input."
                        st.info("Hint only: " + local_hint + " No full solution is shown.")
            elif client is None:
                st.info(offline_code_review(subject, topic, level, task_title, task_description, code))
            else:
                prompt = f"You are a placement coding mentor. Subject: {subject}. Language/area: {subject}. Level: {level}. Topic: {topic}. Problem: {task_title}. Requirements: {task_description}. Student submission:\n{code}\nFirst assess the approach. If incorrect, give only 2-3 hints and the failing edge cases; do NOT reveal the complete solution. If correct, explain why it works and give complexity where relevant. For SQL, check query logic. Do not claim you executed the code."
                with st.spinner("Reviewing your solution..."):
                    feedback = ask_gemini(client, prompt)
                if feedback:
                    if feedback.startswith(("AI review is temporarily unavailable", "Gemini is temporarily busy", "AI review could not connect")):
                        st.info(offline_code_review(subject, topic, level, task_title, task_description, code))
                        st.caption(feedback)
                    else:
                        st.markdown(feedback)
                        save_result(student, subject, topic, 0, 1, level, notes="Solution reviewed by AI")
                else:
                    st.info(offline_code_review(subject, topic, level, task_title, task_description, code))

# ----------------------------- AI MOCK INTERVIEW ----------------------
elif page == "AI Mock Interview":
    st.title("🎤 AI Mock Interview")
    st.write("Practice HR and technical interviews, answer by typing or voice, and receive feedback on strengths and improvements.")
    st.caption("Choose a category. The Mixed Placement Interview contains 50 questions across HR, programming, DSA, DBMS, OS, networks, AI, and projects.")

    interview_type = st.selectbox("Interview category", list(INTERVIEW_QUESTIONS.keys()))
    questions = INTERVIEW_QUESTIONS[interview_type]
    interview_key = "interview_state"
    if (
        interview_key not in st.session_state
        or st.session_state[interview_key].get("type") != interview_type
    ):
        st.session_state[interview_key] = {
            "type": interview_type,
            "index": 0,
            "feedback": None,
            "answer": "",
            "completed": [],
        }

    state = st.session_state[interview_key]
    idx = state["index"]

    if idx >= len(questions):
        st.success("You have completed this interview question set!")
        if state["completed"]:
            scores = [x["score"] for x in state["completed"]]
            st.metric("Average AI score", f"{sum(scores) / len(scores):.1f}/10")
            st.dataframe(pd.DataFrame(state["completed"]), use_container_width=True, hide_index=True)
        if st.button("Restart interview"):
            st.session_state[interview_key] = {
                "type": interview_type, "index": 0, "feedback": None,
                "answer": "", "completed": [],
            }
            st.rerun()
    else:
        question = questions[idx]
        st.progress(idx / len(questions))
        st.caption(f"Question {idx + 1} of {len(questions)}")
        st.subheader(question)

        if st.button("🔊 Read question aloud"):
            if gTTS is None:
                st.error("Text-to-speech package is missing. Install gTTS.")
            else:
                try:
                    audio_buffer = io.BytesIO()
                    gTTS(text=question, lang="en").write_to_fp(audio_buffer)
                    st.audio(audio_buffer.getvalue(), format="audio/mp3")
                except Exception as exc:
                    st.error(f"Could not create speech audio: {exc}")

        answer_widget_key = f"interview_answer_{interview_type}_{idx}"
        pending_transcript_key = f"pending_transcript_{interview_type}_{idx}"
        # Apply voice transcription before creating the text_area widget.
        if pending_transcript_key in st.session_state:
            st.session_state[answer_widget_key] = st.session_state.pop(pending_transcript_key)

        if answer_widget_key not in st.session_state:
            st.session_state[answer_widget_key] = state.get("answer", "")

        typed_answer = st.text_area(
            "Type your answer",
            height=150,
            key=answer_widget_key,
            placeholder="Answer as you would in a real interview...",
        )
        state["answer"] = typed_answer

        if hasattr(st, "audio_input"):
            recording = st.audio_input("Or record your answer (if your browser supports it)")
        else:
            recording = None
            st.caption("Voice recording needs a newer Streamlit version. You can still type your answer.")

        if st.button("Convert recording to text", key=f"transcribe_{idx}"):
            if recording is None:
                st.warning("Record an answer first, or type your answer above.")
            elif sr is None:
                st.error("SpeechRecognition is not installed. Install it using the command below.")
            else:
                try:
                    audio_bytes = recording.getvalue()
                    audio_file = io.BytesIO(audio_bytes)
                    with sr.AudioFile(audio_file) as source:
                        audio_data = sr.Recognizer().record(source)
                    transcript = sr.Recognizer().recognize_google(audio_data)
                    # Do not modify the text_area's session-state key after it is instantiated.
                    st.session_state[pending_transcript_key] = transcript
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not transcribe this recording: {exc}")

        if st.button("Evaluate answer", type="primary", key=f"evaluate_{idx}"):
            final_answer = state.get("answer", "").strip()
            if not final_answer:
                st.warning("Please type or transcribe your answer first.")
            else:
                prompt = f"""
You are an encouraging but honest placement interviewer.
Interview category: {interview_type}
Question: {question}
Candidate answer: {final_answer}
Evaluate this answer for a college placement interview.
Return valid JSON only, with these keys:
score (integer 1 to 10), feedback (short paragraph), strengths (array of strings),
improvements (array of strings), sample_answer (short model answer),
follow_up (one relevant next question).
Be fair. For HR questions, value clarity, specificity, and authenticity.
For technical questions, value correctness and examples.
"""
                raw = None
                if client is not None:
                    with st.spinner("Evaluating your answer..."):
                        raw = ask_gemini(client, prompt)

                if not raw or raw.startswith("Gemini request failed") or raw == "The AI returned an empty response.":
                    result = offline_interview_evaluation(interview_type, question, final_answer)
                    st.info("Gemini is unavailable or over quota. Showing basic offline feedback; it does not verify technical correctness.")
                else:
                    try:
                        cleaned = raw.strip()
                        if cleaned.startswith("```"):
                            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
                        result = json.loads(cleaned)
                    except Exception:
                        result = {
                            "score": 5,
                            "feedback": raw or "The AI did not return a readable evaluation.",
                            "strengths": [],
                            "improvements": ["Try giving a clear, structured answer with a specific example."],
                            "sample_answer": "",
                            "follow_up": "Can you give a practical example?",
                        }

                # Save and display BOTH AI and offline evaluations.
                try:
                    score = max(1, min(10, int(result.get("score", 5))))
                except (TypeError, ValueError):
                    score = 5
                state["feedback"] = result
                state["feedback"]["score"] = score
                state["completed"] = [x for x in state["completed"] if x["question"] != question]
                state["completed"].append({
                    "question": question,
                    "score": score,
                    "feedback": str(result.get("feedback", ""))[:250],
                })
                save_result(student, "Interview", interview_type, score, 10, "", notes=question)
                st.rerun()

        if state.get("feedback"):
            result = state["feedback"]
            st.divider()
            st.subheader(f"AI feedback · {result.get('score', 5)}/10")
            st.write(result.get("feedback", ""))
            if result.get("strengths"):
                st.markdown("**Strengths**")
                for item in result["strengths"]:
                    st.write("✅ " + str(item))
            if result.get("improvements"):
                st.markdown("**How to improve**")
                for item in result["improvements"]:
                    st.write("• " + str(item))
            if result.get("sample_answer"):
                with st.expander("See a sample answer"):
                    st.write(result["sample_answer"])
            if result.get("follow_up"):
                st.markdown("**Possible follow-up question:** " + str(result["follow_up"]))

            if st.button("Next interview question", type="primary", key=f"next_{idx}"):
                state["index"] += 1
                state["feedback"] = None
                state["answer"] = ""
                st.rerun()


# ----------------------------- HISTORY --------------------------------
elif page == "Progress History":
    st.title("📈 Progress History")
    st.caption("Understand your performance, identify weak topics, and decide what to practise next.")
    df = load_results(student)

    if df.empty:
        st.info("No results saved yet. Complete an aptitude, reasoning, coding, or mock interview session to see your progress here.")
    else:
        df["percentage"] = df.apply(lambda r: score_percent(r["score"], r["total"]), axis=1)
        df["percentage"] = pd.to_numeric(df["percentage"], errors="coerce").fillna(0).clip(0, 100)

        # Latest overall summary
        overall = round(float(df["percentage"].mean()), 1)
        total_sessions = len(df)
        latest_date = str(df.iloc[0]["created_at"])
        m1, m2, m3 = st.columns(3)
        m1.metric("Overall average", f"{overall}%")
        m2.metric("Practice sessions", total_sessions)
        m3.metric("Latest activity", latest_date[:10] if latest_date else "—")

        st.subheader("📊 Performance by category")
        category_df = (
            df.groupby("category", as_index=False)
            .agg(score=("score", "sum"), total=("total", "sum"), sessions=("category", "count"))
        )
        category_df["percentage"] = category_df.apply(
            lambda row: score_percent(row["score"], row["total"]), axis=1
        )
        category_df = category_df.sort_values("percentage", ascending=True)

        # Readable chart with category labels and actual percentages
        chart_data = category_df.set_index("category")[["percentage"]]
        st.bar_chart(chart_data, y="percentage", y_min=0, y_max=100)
        for _, row in category_df.iterrows():
            pct = float(row["percentage"])
            if pct < 40:
                status = "Needs improvement — start here"
            elif pct < 60:
                status = "Keep practising"
            elif pct < 80:
                status = "Good progress"
            else:
                status = "Excellent"
            st.write(f"**{row['category']} — {pct:.1f}%** · {status} · {int(row['sessions'])} session(s)")
            st.progress(min(max(pct / 100, 0.0), 1.0), text=f"{pct:.1f}% completed score")

        st.subheader("🎯 Topics that need attention")
        topic_df = (
            df.groupby(["category", "topic"], as_index=False)
            .agg(score=("score", "sum"), total=("total", "sum"), sessions=("topic", "count"))
        )
        topic_df["percentage"] = topic_df.apply(
            lambda row: score_percent(row["score"], row["total"]), axis=1
        )
        weak_topics = topic_df[topic_df["percentage"] < 60].sort_values("percentage", ascending=True)

        if weak_topics.empty:
            st.success("Great work! No topic is currently below 60%. Try medium/hard questions to keep improving.")
        else:
            st.warning("Prioritise these topics first. The suggestions are based on your saved scores (below 60%).")
            for _, row in weak_topics.head(8).iterrows():
                pct = float(row["percentage"])
                if pct < 40:
                    advice = "Revise the basic concepts and formulas, then solve 5 easy questions before moving up a level."
                else:
                    advice = "Review incorrect answers and practise 10 more questions from this topic."
                with st.container(border=True):
                    left, right = st.columns([3, 1])
                    left.markdown(f"**{row['category']} → {row['topic']}**")
                    left.write(advice)
                    right.metric("Topic score", f"{pct:.1f}%")
                    st.progress(min(max(pct / 100, 0.0), 1.0))

        with st.expander("View topic-wise scores"):
            view_topics = topic_df.sort_values("percentage", ascending=True).copy()
            view_topics["percentage"] = view_topics["percentage"].map(lambda value: f"{value:.1f}%")
            st.dataframe(view_topics, use_container_width=True, hide_index=True)

        st.subheader("📅 Score trend")
        trend = df.iloc[::-1].copy()
        trend["session"] = range(1, len(trend) + 1)
        st.line_chart(trend.set_index("session")[["percentage"]], y="percentage", y_min=0, y_max=100)

        with st.expander("View all saved sessions"):
            st.dataframe(df, use_container_width=True, hide_index=True)

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download history as CSV",
            data=csv_data,
            file_name="placement_progress.csv",
            mime="text/csv",
        )

st.sidebar.divider()
st.sidebar.caption("Your app password is stored locally as a salted hash. Configure GEMINI_API_KEY privately to enable Gemini features.")
