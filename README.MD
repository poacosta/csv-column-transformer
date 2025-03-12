# CSV Column Transformer: My Battle-Tested Solution for Data Wrangling 🛠️

Let's face it: dealing with massive CSVs is nobody's idea of a good time. After one too many memory errors and
coffee-fueled debugging sessions, I created this toolkit to solve that perpetual data engineer headache - transforming
just *one column* in a sea of data without breaking everything else.

## The Problem (We've All Been There)

Picture this: You've got a 400k-row CSV behemoth. Marketing needs all those product images paths changed, but only the
first segment removed. The deadline? Yesterday, naturally. Your first attempt with Excel crashes spectacularly, and that
quick pandas script is now consuming more RAM than Chrome with 50 tabs open.

## My Sanity-Preserving Solution

This toolkit offers two complementary approaches:

1. **ColumnTransformer Class**: A clean, object-oriented API for the Python purists
2. **CSV Column Transformer CLI**: For when you need to process industrial-sized datasets without setting your laptop on
   fire

Both maintain your CSV's structure while surgically altering just the values you need to change - like performing
keyhole surgery instead of open-heart.

## ✨ What It Actually Does

- **Path & URL Wizardry**: Intelligently strips first segments from both URLs (
  `https://example.com/products/widget.jpg` → `/products/widget.jpg`) and simple paths (
  `3760147560023/PROD59315.jpg` → `PROD59315.jpg`)
- **Text Transformations**: Add prefixes/suffixes because marketing decided everything needs "NEW-" in front of it...
  again
- **Type Juggling**: Convert between strings and numbers because someone exported that price column as text (we've all
  been there)
- **Memory-Friendly Processing**: Handles your massive files in bite-sized chunks that won't cause your system to have
  an existential crisis
- **Progress Tracking**: Tells you exactly how much longer you'll be waiting so you can plan your coffee breaks
  accordingly

## 🚀 Quickstart (Because Who Reads Documentation?)

```bash
# Clone it (or just copy the files, I won't judge)
git clone https://github.com/poacosta/csv-column-transformer.git
cd csv-column-transformer

# Install dependencies (just pandas, we're not building a rocket here)
pip install pandas
```

### For the Object-Oriented Enthusiasts

```python
from column_transformer import ColumnTransformer, remove_first_segment

# Point it at your files
transformer = ColumnTransformer("massive_product_data.csv", "fixed_product_data.csv")

# Do the thing
transformer.load_csv()
transformer.transform_column("image_path", remove_first_segment)
transformer.save_csv()

# Go get coffee while it runs
```

### For Command Line Warriors

```bash
# Basic path transformation 
python csv_column_transformer.py products.csv output.csv image_path remove_first_segment

# Add that marketing prefix
python csv_column_transformer.py products.csv output.csv product_name add_prefix --prefix "NEW-"

# Fix that numeric data someone exported as strings
python csv_column_transformer.py products.csv output.csv price to_numeric
```

## 🔮 Real-World Examples (Because Theory is Boring)

### Before & After: Path Processing

| Input                                    | Output                    | Marketing's Reaction            |
|------------------------------------------|---------------------------|---------------------------------|
| `3760147560023/PROD59315.jpg`            | `PROD59315.jpg`           | Surprisingly positive           |
| `https://site.com/products/item123.html` | `/products/item123.html`  | "Can you make it blue?"         |
| `category/subcategory/product.jpg`       | `subcategory/product.jpg` | "Perfect! Wait, can we also..." |

### Code for Common Scenarios

```python
# The "marketing changed their mind again" scenario
transformer.transform_column("product_name", lambda s: add_prefix(s, "SUMMER-"))

# The "fix the data export" scenario 
transformer.transform_column("price_column", cast_to_numeric)

# The "everything is on sale" scenario
transformer.transform_column("product_code", lambda s: add_suffix(s, "-SALE"))
```

## 🧠 For the Power Users

### Chain Everything (Like a Functional Programming Guru)

```python
(transformer
 .load_csv(low_memory=False)
 .transform_column("image_path", remove_first_segment)
 .transform_column("product_name", lambda s: add_prefix(s, "2025-"))
 .transform_column("price", cast_to_string)
 .save_csv())
```

## 🚦 Real Talk: Performance Considerations

Let me save you some pain with what I've learned:

- **Under 100k rows**: Use the standard `ColumnTransformer` - simple, clean, gets the job done
- **100k-10M rows**: Definitely use `csv_column_transformer.py` with chunking
- **Over 10M rows**: Maybe reconsider your life choices... or at least adjust that chunk size

Processing speed feels like it varies wildly based on transformation complexity.

## 🔍 Under the Hood (For the Technically Curious)

This isn't rocket science (though it sometimes felt like it during testing). The tools use:

1. Pandas for the heavy lifting (because reinventing that wheel would be masochism)
2. Chunked processing to keep memory usage from exploding
3. Efficient series operations instead of slow row-by-row iteration
4. Targeted column transformations that leave everything else untouched

## 📋 Minimum Requirements

- Python 3.8+ (might work on earlier versions, but who's still using those?)
- pandas
- A CSV file that's causing you problems
- Coffee (optional but recommended)

## 🚨 When Things Go Wrong (Because They Will)

- **Memory Errors**: Decrease that chunk size, maybe close Chrome for once
- **Slow Processing**: Check if you're trying to transform the entire CSV instead of just one column
- **CSV Format Errors**: CSVs are never as simple as they seem - check encodings, delimiters, etc.
- **Missing Values**: Don't worry, NaN/None values stay that way by default

## 🔜 Future Improvements (When I Find The Time)

- Regex-based transformations for when simple replacements aren't enough
- Multi-column operations for the truly ambitious
- Parallel processing for the impatient
- Config files for complex transformation recipes

---

Created while muttering "there has to be a better way to do this" at 2 AM. If this saves you even one late night
debugging session, my work here is done.