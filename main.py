import pandas as pd
import anthropic
import re
import time
from typing import Dict, List, Tuple
import json
from datetime import datetime
import os


class MarketResearchDataCleaner:
    def __init__(
        self, api_key: str, brand_list: List[str] = None, brand_file: str = None
    ):
        """
        Initialize the cleaner with Claude API.

        Args:
            api_key: Anthropic API key
            brand_list: Optional list of brands to check for
            brand_file: Optional path to file containing brand list (one per line)
        """
        self.client = anthropic.Anthropic(api_key=api_key)

        # Profanity list (expandable)
        self.profanity_patterns = [
            r"\bf[*u@]ck",
            r"\bsh[*i!]t",
            r"\bdam[nm]",
            r"\bhell\b",
            r"\bcrap",
            r"\bbastard",
            r"\bpiss",
            r"\bass\b",
        ]

        # No experience indicators
        self.no_experience_phrases = [
            "no experience",
            "none",
            "n/a",
            "nil",
            "not applicable",
            "blank",
            "empty",
        ]

        # Initialize brand list
        self.common_brands = []
        self.brand_variations = {}

        # Load brands from file if provided
        if brand_file and os.path.exists(brand_file):
            with open(brand_file, "r", encoding="utf-8") as f:
                self.common_brands = [
                    line.strip().lower() for line in f if line.strip()
                ]
        # Use provided brand list
        elif brand_list:
            self.common_brands = [brand.lower() for brand in brand_list]

        # Common touchpoint keywords (universal across categories)
        self.touchpoint_keywords = {
            "tv": [
                "tv",
                "television",
                "advert",
                "commercial",
                "channel",
                "watching",
                "programme",
                "show",
            ],
            "online": [
                "online",
                "website",
                "internet",
                "digital",
                "web",
                "browse",
                "search",
                "click",
            ],
            "social media": [
                "social",
                "facebook",
                "instagram",
                "twitter",
                "tiktok",
                "linkedin",
                "feed",
                "post",
                "story",
            ],
            "outdoor": [
                "outdoor",
                "billboard",
                "poster",
                "bus stop",
                "street",
                "ooh",
                "outside",
            ],
            "in a shop": [
                "shop",
                "store",
                "retail",
                "supermarket",
                "tesco",
                "sainsbury",
                "asda",
                "shelf",
                "aisle",
            ],
            "in a bar/pub/club": [
                "bar",
                "pub",
                "club",
                "drink",
                "tap",
                "bartender",
                "menu",
            ],
            "in a restaurant/cafe": [
                "restaurant",
                "cafe",
                "menu",
                "waiter",
                "table",
                "dining",
                "meal",
            ],
            "at home": ["home", "house", "flat", "kitchen", "fridge", "living room"],
            "conversation": [
                "conversation",
                "talk",
                "discuss",
                "chat",
                "mention",
                "told",
                "said",
            ],
            "audio": ["radio", "podcast", "audio", "listen", "spotify", "music"],
            "newspaper or magazine": [
                "newspaper",
                "magazine",
                "article",
                "print",
                "read",
                "page",
            ],
            "cinema": ["cinema", "movie", "film", "screen", "theater"],
            "sponsorship": ["sponsor", "sponsored", "partnership", "event"],
            "me drinking/purchasing": [
                "bought",
                "purchase",
                "drink",
                "drank",
                "tried",
                "taste",
            ],
            "someone else drinking/purchasing": [
                "they bought",
                "friend had",
                "someone drinking",
                "saw someone",
                "they ordered",
            ],
        }

    def extract_brands_from_dataset(self, df: pd.DataFrame) -> List[str]:
        """Extract unique brands from the dataset."""
        if "brand" in df.columns:
            brands = df["brand"].dropna().unique()
            return [str(brand).lower() for brand in brands]
        return []

    def generate_brand_variations(self, brand: str) -> List[str]:
        """Generate common variations of a brand name."""
        brand_lower = brand.lower()
        variations = [brand_lower]

        # Handle abbreviations and common patterns
        if " & " in brand_lower:
            variations.append(brand_lower.replace(" & ", " and "))
            variations.append(brand_lower.replace(" & ", " "))

        if " and " in brand_lower:
            variations.append(brand_lower.replace(" and ", " & "))
            variations.append(brand_lower.replace(" and ", " "))

        # Handle possessives
        if "'s" in brand_lower:
            variations.append(brand_lower.replace("'s", ""))
            variations.append(brand_lower.replace("'s", "s"))

        # Remove special characters for matching
        clean_brand = re.sub(r"[^\w\s]", "", brand_lower)
        if clean_brand != brand_lower:
            variations.append(clean_brand)

        # Handle multi-word brands (e.g., "Stella Artois" -> "Stella")
        words = brand_lower.split()
        if len(words) > 1:
            variations.append(words[0])  # First word only
            variations.append(words[-1])  # Last word only

        return list(set(variations))  # Remove duplicates

    def check_profanity(self, text: str) -> bool:
        """Check if text contains profanity."""
        if pd.isna(text) or text is None or not text:
            return False
        text_str = str(text)
        text_lower = text_str.lower()
        return any(
            re.search(pattern, text_lower) for pattern in self.profanity_patterns
        )

    def check_gibberish(self, text: str) -> Tuple[bool, str]:
        """Basic gibberish detection."""
        if pd.isna(text) or text is None:
            return True, "No text provided"

        text_str = str(text)
        if len(text_str) < 5:
            return True, "Text too short"

        # Check for repeated characters
        if re.search(r"(.)\1{4,}", text_str):
            return True, "Excessive character repetition"

        # Check word structure
        words = text_str.split()
        if not words:
            return True, "No words found"

        # Check for words without vowels (excluding common abbreviations)
        no_vowel_words = [
            w for w in words if len(w) > 4 and not re.search(r"[aeiouAEIOU]", w)
        ]
        if len(no_vowel_words) > len(words) * 0.3:
            return True, "Too many words without vowels"

        # Check for excessive numbers/symbols
        alpha_chars = len(re.findall(r"[a-zA-Z]", text_str))
        if alpha_chars == 0:
            return True, "No alphabetic characters"
        alpha_ratio = alpha_chars / len(text_str)
        if alpha_ratio < 0.5:
            return True, "Too many non-alphabetic characters"

        return False, ""

    def check_no_experience(self, text: str) -> bool:
        """Check if response indicates no real experience."""
        if pd.isna(text) or text is None or not text:
            return True
        text_str = str(text)
        text_lower = text_str.lower()
        return any(phrase in text_lower for phrase in self.no_experience_phrases)

    def basic_brand_check(self, brand: str, story: str) -> int:
        """Basic brand matching without Claude API."""
        if not story or not brand:
            return 50

        story_lower = story.lower()
        brand_lower = brand.lower()

        # Generate variations for the reported brand
        brand_variations = self.generate_brand_variations(brand)

        # Check if any variation of the reported brand is mentioned
        for variation in brand_variations:
            # Use word boundaries for more accurate matching
            if re.search(r"\b" + re.escape(variation) + r"\b", story_lower):
                return 100

        # Check if any other brand from the dataset is mentioned
        mentioned_other_brand = False
        for other_brand in self.common_brands:
            if other_brand != brand_lower:
                other_variations = self.generate_brand_variations(other_brand)
                for variation in other_variations:
                    if re.search(r"\b" + re.escape(variation) + r"\b", story_lower):
                        # Make sure it's not part of the expected brand
                        if not any(
                            var in other_brand or other_brand in var
                            for var in brand_variations
                        ):
                            mentioned_other_brand = True
                            break
                if mentioned_other_brand:
                    break

        if mentioned_other_brand:
            return -100  # Different brand mentioned

        # No brand mentioned
        return 50

    def basic_touchpoint_check(self, touchpoint: str, story: str) -> int:
        """Basic touchpoint matching without Claude API."""
        if not story or not touchpoint:
            return 50

        story_lower = story.lower()
        touchpoint_lower = touchpoint.lower()

        # Find which category the reported touchpoint belongs to
        reported_category = None
        for tp_category, keywords in self.touchpoint_keywords.items():
            if any(keyword in touchpoint_lower for keyword in keywords):
                reported_category = tp_category
                break

        # Check what touchpoint is described in the story
        described_categories = []
        for tp_category, keywords in self.touchpoint_keywords.items():
            if any(keyword in story_lower for keyword in keywords):
                described_categories.append(tp_category)

        # Scoring logic
        if not described_categories:
            # No clear touchpoint mentioned
            return 50
        elif reported_category and reported_category in described_categories:
            # Correct touchpoint mentioned
            return 100
        else:
            # Different touchpoint mentioned
            return -100

    def analyze_with_claude(self, row: pd.Series) -> Dict:
        """Use Claude to analyze a single row for quality issues."""
        brand = row.get("brand", "")
        touchpoint = row.get("touchpoint", "")
        story = row.get("text", "")

        # Handle NaN/float values
        if pd.isna(brand) or brand is None:
            brand = ""
        else:
            brand = str(brand)

        if pd.isna(touchpoint) or touchpoint is None:
            touchpoint = ""
        else:
            touchpoint = str(touchpoint)

        if pd.isna(story) or story is None:
            story = ""
        else:
            story = str(story)

        # Skip if essential fields are missing
        if not brand or not touchpoint or not story:
            return {
                "brand_match_score": 0,
                "touchpoint_match_score": 0,
                "quality_score": 0,
                "issues": ["Missing essential fields"],
                "recommendation": "Remove - incomplete data",
            }

        # Clean story text to prevent JSON issues
        story_cleaned = story.replace('"', "'").replace("\n", " ").replace("\r", " ")
        if len(story_cleaned) > 500:  # Limit story length for API
            story_cleaned = story_cleaned[:500] + "..."

        # Create brand list context
        if self.common_brands:
            # Only send a subset to save tokens
            brands_sample = (
                self.common_brands[:30]
                if len(self.common_brands) > 30
                else self.common_brands
            )
            brands_context = (
                f"Brands in this dataset include: {', '.join(brands_sample)}, etc."
            )
        else:
            brands_context = "No specific brand list provided."

        prompt = f"""Analyze this market research response for data quality issues.

Brand reported: {brand}
Touchpoint reported: {touchpoint}
Experience description: {story_cleaned}

{brands_context}

Please evaluate using this scoring system:

BRAND SCORING:
- Score 100: The reported brand "{brand}" is explicitly mentioned in the "{story_cleaned}" column
- Score 50: The reported brand "{brand}" is not explicitly mentioned and / or no brand is mentioned at all in the "{story_cleaned}" (this is common)
- Score -100: A DIFFERENT brand is mentioned in the "{story_cleaned}" column instead of "{brand}" (major red flag)

TOUCHPOINT SCORING:
- Score 100: The reported touchpoint "{touchpoint}" is explicitly mentioned in the "{story_cleaned}" column
- Score 50: The reported touchpoint "{touchpoint}" is not explicitly mentioned and / or no touchpoint is mentioned at all in the "{story_cleaned}" (this is common)
- Score -100: A DIFFERENT touchpoint is mentioned in the "{story_cleaned}" column instead of "{touchpoint}" (major red flag)

QUALITY: Rate the overall coherence and genuineness of the experience (0-100)

Respond ONLY with valid JSON:
{{
    "brand_match_score": <100, 50, or -100>,
    "touchpoint_match_score": <100, 50, or -100>,
    "quality_score": <0-100>,
    "brand_mentioned": "<actual brand mentioned or 'none'>",
    "touchpoint_detected": "<actual touchpoint described or 'unclear'>",
    "issues": [<list of specific issues found>],
    "summary": "<brief explanation>"
}}"""

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",  # Using Haiku for efficiency
                max_tokens=300,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )

            # Get response text
            response_text = response.content[0].text.strip()

            # Try to extract JSON if it's wrapped in other text
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                response_text = response_text[json_start:json_end]

            # Parse Claude's response
            result = json.loads(response_text)

            # Validate expected fields
            required_fields = [
                "brand_match_score",
                "touchpoint_match_score",
                "quality_score",
            ]
            for field in required_fields:
                if field not in result:
                    result[field] = 50  # Default score

            # Ensure scores are in valid range
            if result["brand_match_score"] not in [100, 50, -100]:
                result["brand_match_score"] = 50
            if result["touchpoint_match_score"] not in [100, 50, -100]:
                result["touchpoint_match_score"] = 50

            return result

        except json.JSONDecodeError as e:
            print(f"JSON parsing error for row {row.name}: {e}")
            if "response_text" in locals():
                print(f"Response was: {response_text[:200]}...")
            return {
                "brand_match_score": 50,
                "touchpoint_match_score": 50,
                "quality_score": 50,
                "issues": ["Claude response parsing error"],
                "summary": "Could not parse Claude response",
            }
        except Exception as e:
            print(f"Error analyzing row {row.name}: {e}")
            return {
                "brand_match_score": 50,
                "touchpoint_match_score": 50,
                "quality_score": 50,
                "issues": ["Analysis error"],
                "summary": f"Error: {str(e)}",
            }

    def score_row(self, row: pd.Series, use_claude: bool = True) -> Dict:
        """Score a single row combining all checks."""
        # Handle potential float/NaN values
        story = row.get("text", "")
        if pd.isna(story) or story is None:
            story = ""
        else:
            story = str(story)  # Convert to string to handle any type

        brand = row.get("brand", "")
        if pd.isna(brand) or brand is None:
            brand = ""
        else:
            brand = str(brand)

        touchpoint = row.get("touchpoint", "")
        if pd.isna(touchpoint) or touchpoint is None:
            touchpoint = ""
        else:
            touchpoint = str(touchpoint)

        scores = {
            "row_index": row.name,
            "brand": brand,
            "touchpoint": touchpoint,
            "story_preview": story[:100] + "..." if len(story) > 100 else story,
            "has_profanity": False,
            "is_gibberish": False,
            "gibberish_reason": "",
            "is_no_experience": False,
            "story_length": len(story) if story else 0,
            "brand_match_score": 50,
            "touchpoint_match_score": 50,
            "quality_score": 50,
            "issues": [],
            "summary": "Not analyzed",
        }

        # Basic checks
        scores["has_profanity"] = self.check_profanity(story)
        scores["is_gibberish"], scores["gibberish_reason"] = self.check_gibberish(story)
        scores["is_no_experience"] = self.check_no_experience(story)

        # Claude analysis (optional for speed)
        if use_claude and story and len(story) > 10:
            claude_analysis = self.analyze_with_claude(row)
            scores.update(claude_analysis)
        else:
            # Basic analysis without Claude
            scores["brand_match_score"] = self.basic_brand_check(brand, story)
            scores["touchpoint_match_score"] = self.basic_touchpoint_check(
                touchpoint, story
            )
            scores["quality_score"] = (
                0
                if scores["is_gibberish"]
                else (50 if scores["is_no_experience"] else 70)
            )
            scores["summary"] = "Basic analysis only"

        # Calculate overall score
        penalty = 0
        if scores["has_profanity"]:
            penalty += 30
            scores["issues"].append("Contains profanity")
        if scores["is_gibberish"]:
            penalty += 40
            scores["issues"].append(f"Gibberish: {scores['gibberish_reason']}")
        if scores["is_no_experience"]:
            penalty += 30
            scores["issues"].append("No real experience described")
        if scores["story_length"] < 20:
            penalty += 20
            scores["issues"].append("Very short response")

        # Handle brand/touchpoint mismatches
        brand_score = scores.get("brand_match_score", 50)
        touchpoint_score = scores.get("touchpoint_match_score", 50)

        if brand_score == -100:
            scores["issues"].append("CRITICAL: Different brand mentioned")
        if touchpoint_score == -100:
            scores["issues"].append("CRITICAL: Different touchpoint described")

        # Overall score calculation
        # For mismatches (negative scores), we want to heavily penalize
        if brand_score < 0 or touchpoint_score < 0:
            # If there's a mismatch, start with a very low base score
            base_score = 0
        else:
            # Normal calculation for non-negative scores
            base_score = (
                brand_score + touchpoint_score + scores.get("quality_score", 50)
            ) / 3

        scores["overall_score"] = max(0, base_score - penalty)

        # Recommendation based on new scoring
        if brand_score < 0 or touchpoint_score < 0:
            scores["recommendation"] = "Remove - brand/touchpoint mismatch"
        elif scores["overall_score"] < 30:
            scores["recommendation"] = "Remove - poor quality"
        elif scores["overall_score"] < 60:
            scores["recommendation"] = "Review - potential issues"
        else:
            scores["recommendation"] = "Keep - appears valid"

        return scores

    def process_dataset(
        self,
        df: pd.DataFrame,
        use_claude: bool = True,
        sample_size: int = None,
        debug: bool = False,
        auto_extract_brands: bool = True,
    ) -> pd.DataFrame:
        """
        Process entire dataset and return scored results.

        Args:
            df: Input dataframe
            use_claude: Whether to use Claude API
            sample_size: Number of rows to process (None = all)
            debug: Show debug information
            auto_extract_brands: Automatically extract brands from dataset
        """
        print(f"Processing {len(df)} rows...")

        # Auto-extract brands from dataset if enabled and no brands loaded
        if auto_extract_brands and not self.common_brands:
            print("🔍 Extracting brands from dataset...")
            dataset_brands = self.extract_brands_from_dataset(df)
            self.common_brands = dataset_brands
            print(f"   Found {len(dataset_brands)} unique brands")

            # Generate variations for all brands
            print("   Generating brand variations...")
            for brand in dataset_brands:
                variations = self.generate_brand_variations(brand)
                self.brand_variations[brand] = variations

        # Sample if requested (for testing)
        if sample_size and sample_size < len(df):
            df_sample = df.sample(n=sample_size, random_state=42)
            print(f"Using sample of {sample_size} rows")
        else:
            df_sample = df

        results = []
        error_rows = []

        for idx, row in df_sample.iterrows():
            if idx % 10 == 0:
                print(f"Processing row {idx}/{len(df_sample)}...")

            try:
                scores = self.score_row(row, use_claude=use_claude)
                results.append(scores)

                # Debug mode: print details for problematic rows
                if debug and "Claude response parsing error" in scores.get(
                    "issues", []
                ):
                    story_val = row.get("text", "")
                    if pd.isna(story_val) or story_val is None:
                        story_preview = ""
                    else:
                        story_str = str(story_val)
                        story_preview = (
                            story_str[:100] if len(story_str) > 100 else story_str
                        )

                    error_rows.append(
                        {
                            "index": idx,
                            "brand": str(row.get("brand", ""))
                            if not pd.isna(row.get("brand"))
                            else "",
                            "touchpoint": str(row.get("touchpoint", ""))
                            if not pd.isna(row.get("touchpoint"))
                            else "",
                            "story_preview": story_preview,
                        }
                    )

            except Exception as e:
                print(f"ERROR processing row {idx}: {e}")
                # Add a default score for failed rows
                story_val = row.get("text", "")
                if pd.isna(story_val) or story_val is None:
                    story_preview = ""
                else:
                    story_str = str(story_val)
                    story_preview = (
                        story_str[:100] if len(story_str) > 100 else story_str
                    )

                results.append(
                    {
                        "row_index": idx,
                        "brand": str(row.get("brand", ""))
                        if not pd.isna(row.get("brand"))
                        else "",
                        "touchpoint": str(row.get("touchpoint", ""))
                        if not pd.isna(row.get("touchpoint"))
                        else "",
                        "story_preview": story_preview,
                        "overall_score": 0,
                        "brand_match_score": 0,
                        "touchpoint_match_score": 0,
                        "quality_score": 0,
                        "has_profanity": False,
                        "is_gibberish": False,
                        "is_no_experience": False,
                        "story_length": 0,
                        "recommendation": "Error - manual review needed",
                        "issues": [f"Processing error: {str(e)}"],
                        "summary": "Processing error",
                    }
                )

            # Rate limiting for Claude API
            if use_claude:
                time.sleep(0.5)  # Adjust based on your API limits

        if debug and error_rows:
            print(f"\n=== ROWS WITH PARSING ERRORS ({len(error_rows)}) ===")
            for err in error_rows[:5]:  # Show first 5
                print(f"\nRow {err['index']}:")
                print(f"Brand: {err['brand']}")
                print(f"Touchpoint: {err['touchpoint']}")
                print(f"Story: {err['story_preview']}...")

        return pd.DataFrame(results)

    def generate_report(self, scored_df: pd.DataFrame) -> str:
        """Generate summary report of findings."""
        # Handle missing values
        scored_df = scored_df.fillna(
            {
                "overall_score": 0,
                "brand_match_score": 50,
                "touchpoint_match_score": 50,
                "quality_score": 0,
                "has_profanity": False,
                "is_gibberish": False,
                "is_no_experience": False,
                "story_length": 0,
                "recommendation": "Unknown",
            }
        )

        report = f"""
Market Research Data Quality Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
=====================================

Total Rows Analyzed: {len(scored_df)}

Quality Distribution:
- High Quality (70-100): {len(scored_df[scored_df["overall_score"] >= 70])} ({len(scored_df[scored_df["overall_score"] >= 70]) / len(scored_df) * 100:.1f}%)
- Medium Quality (30-69): {len(scored_df[(scored_df["overall_score"] >= 30) & (scored_df["overall_score"] < 70)])} ({len(scored_df[(scored_df["overall_score"] >= 30) & (scored_df["overall_score"] < 70)]) / len(scored_df) * 100:.1f}%)
- Low Quality (0-29): {len(scored_df[scored_df["overall_score"] < 30])} ({len(scored_df[scored_df["overall_score"] < 30]) / len(scored_df) * 100:.1f}%)

Brand Matching:
- Brand mentioned correctly: {len(scored_df[scored_df["brand_match_score"] == 100])} ({len(scored_df[scored_df["brand_match_score"] == 100]) / len(scored_df) * 100:.1f}%)
- No brand mentioned: {len(scored_df[scored_df["brand_match_score"] == 50])} ({len(scored_df[scored_df["brand_match_score"] == 50]) / len(scored_df) * 100:.1f}%)
- WRONG brand mentioned: {len(scored_df[scored_df["brand_match_score"] == -100])} ({len(scored_df[scored_df["brand_match_score"] == -100]) / len(scored_df) * 100:.1f}%)

Touchpoint Matching:
- Touchpoint matches: {len(scored_df[scored_df["touchpoint_match_score"] == 100])} ({len(scored_df[scored_df["touchpoint_match_score"] == 100]) / len(scored_df) * 100:.1f}%)
- Touchpoint unclear: {len(scored_df[scored_df["touchpoint_match_score"] == 50])} ({len(scored_df[scored_df["touchpoint_match_score"] == 50]) / len(scored_df) * 100:.1f}%)
- WRONG touchpoint: {len(scored_df[scored_df["touchpoint_match_score"] == -100])} ({len(scored_df[scored_df["touchpoint_match_score"] == -100]) / len(scored_df) * 100:.1f}%)

Other Issues Found:
- Profanity: {scored_df["has_profanity"].sum()} rows
- Gibberish: {scored_df["is_gibberish"].sum()} rows
- No Experience: {scored_df["is_no_experience"].sum()} rows
- Very Short (<20 chars): {len(scored_df[scored_df["story_length"] < 20])} rows

Recommendations:
- Remove: {len(scored_df[scored_df["recommendation"].str.contains("Remove")])} rows
- Review: {len(scored_df[scored_df["recommendation"] == "Review - potential issues"])} rows
- Keep: {len(scored_df[scored_df["recommendation"] == "Keep - appears valid"])} rows

Average Scores:
- Brand Match: {scored_df["brand_match_score"].mean():.1f}
- Touchpoint Match: {scored_df["touchpoint_match_score"].mean():.1f}
- Quality Score: {scored_df["quality_score"].mean():.1f}/100
- Overall Score: {scored_df["overall_score"].mean():.1f}/100
"""
        return report


# Example usage
if __name__ == "__main__":
    # Configuration
    API_KEY = os.environ.get("ANTHROPIC_API_KEY")
    if not API_KEY:
        raise ValueError("Please set the ANTHROPIC_API_KEY environment variable")

    INPUT_FILE = "data_cleaning/cleaning_test_data.csv"
    OUTPUT_FILE = (
        f"data_cleaning/cleaned_data_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv"
    )
    REPORT_FILE = f"data_cleaning/data_quality_report_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.txt"

    # Optional: Brand list file (one brand per line)
    BRAND_FILE = "brands.txt"  # Set to None to auto-extract from data

    # Test mode - set to True to run a quick test
    TEST_MODE = False

    if TEST_MODE:
        print("=== RUNNING IN TEST MODE ===")
        print("Testing basic functionality without processing full dataset...")

        # Create test data
        test_data = pd.DataFrame(
            {
                "brand": ["Nike", "Adidas", "Puma", "Reebok"],
                "touchpoint": ["In a shop", "TV", "Online", "Social media"],
                "text": [
                    "Saw the new Nike shoes in the store display",  # Good match
                    "Watching TV and saw a Puma commercial",  # Wrong brand
                    "Browsing Instagram when I saw an ad",  # No brand mentioned
                    "xxxxxxxxxx",  # Gibberish
                ],
            }
        )

        # Initialize cleaner - auto-extract brands from test data
        cleaner = MarketResearchDataCleaner(API_KEY)
        cleaner.common_brands = cleaner.extract_brands_from_dataset(test_data)

        # Test without Claude API
        print("\nTesting basic checks (no API)...")
        for idx, row in test_data.iterrows():
            scores = cleaner.score_row(row, use_claude=False)
            print(f"\nTest {idx + 1}:")
            print(f"  Brand: {row['brand']} -> Score: {scores['brand_match_score']}")
            print(
                f"  Overall: {scores['overall_score']}, Recommendation: {scores['recommendation']}"
            )

        print("\nTest complete! Set TEST_MODE=False to process actual data.")
        exit(0)

    # Normal processing
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)

    # Initialize cleaner with different options:

    # Option 1: Auto-extract brands from dataset (default)
    cleaner = MarketResearchDataCleaner(API_KEY)

    # Option 2: Load brands from file
    # cleaner = MarketResearchDataCleaner(API_KEY, brand_file=BRAND_FILE)

    # Option 3: Provide brand list directly
    # custom_brands = ['nike', 'adidas', 'puma', 'reebok', 'new balance']
    # cleaner = MarketResearchDataCleaner(API_KEY, brand_list=custom_brands)

    # Process dataset
    scored_df = cleaner.process_dataset(
        df,
        use_claude=True,  # Set to True to use Claude API
        sample_size=30,  # Remove or set to None to process all rows
        debug=True,  # Set to True to see error details
        auto_extract_brands=True,  # Auto-extract brands from data
    )

    # Merge scores with original data
    # Drop duplicate columns from scored_df to avoid conflicts
    columns_to_drop = ["brand", "touchpoint", "story_preview"]
    scored_df_clean = scored_df.drop(
        columns=[col for col in columns_to_drop if col in scored_df.columns],
        errors="ignore",
    )

    final_df = df.merge(
        scored_df_clean,
        left_index=True,
        right_on="row_index",
        how="inner",
        suffixes=("", "_score"),
    )

    # Sort by overall score (worst first for easy review)
    final_df = final_df.sort_values("overall_score", ascending=True)

    # Save results
    print(f"\nSaving results to {OUTPUT_FILE}...")
    final_df.to_csv(OUTPUT_FILE, index=False)

    # Generate and save report
    report = cleaner.generate_report(scored_df)
    print(report)

    with open(REPORT_FILE, "w") as f:
        f.write(report)

    # Save brand list for reference
    if cleaner.common_brands:
        with open("extracted_brands.txt", "w", encoding="utf-8") as f:
            for brand in sorted(set(cleaner.common_brands)):
                f.write(f"{brand}\n")
        print("\nExtracted brands saved to extracted_brands.txt")

    print(f"\nReport saved to {REPORT_FILE}")
    print(f"Scored data saved to {OUTPUT_FILE}")

    # Show worst examples
    print("\n=== WORST QUALITY EXAMPLES ===")
    worst_rows = scored_df.sort_values("overall_score", ascending=True).head(5)
    for _, row in worst_rows.iterrows():
        print(f"\nScore: {row['overall_score']:.1f}")
        print(
            f"Brand: {row.get('brand', 'N/A')}, Touchpoint: {row.get('touchpoint', 'N/A')}"
        )
        print(
            f"Brand Score: {row.get('brand_match_score', 'N/A')}, Touchpoint Score: {row.get('touchpoint_match_score', 'N/A')}"
        )
        print(f"Issues: {row.get('issues', [])}")
        print(f"Story: {row.get('story_preview', 'N/A')}")
        print(f"Recommendation: {row.get('recommendation', 'N/A')}")

    # Show brand mismatches specifically
    brand_mismatches = scored_df[scored_df["brand_match_score"] == -100]
    if len(brand_mismatches) > 0:
        print("\n=== BRAND MISMATCHES ===")
        print(f"Found {len(brand_mismatches)} cases where wrong brand was mentioned:")
        for _, row in brand_mismatches.head(3).iterrows():
            print(f"\nRow {row.get('row_index', 'N/A')}:")
            print(f"Expected: {row.get('brand', 'N/A')}")
            print(f"Story: {row.get('story_preview', 'N/A')}")

    # Show touchpoint mismatches
    touchpoint_mismatches = scored_df[scored_df["touchpoint_match_score"] == -100]
    if len(touchpoint_mismatches) > 0:
        print("\n=== TOUCHPOINT MISMATCHES ===")
        print(
            f"Found {len(touchpoint_mismatches)} cases where wrong touchpoint was described:"
        )
        for _, row in touchpoint_mismatches.head(3).iterrows():
            print(f"\nRow {row.get('row_index', 'N/A')}:")
            print(f"Expected: {row.get('touchpoint', 'N/A')}")
            print(f"Story: {row.get('story_preview', 'N/A')}")
