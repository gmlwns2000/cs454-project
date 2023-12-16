import re
from natural import *

def detect_comment_or_next_line(line):
  if re.match(r'^[\+\-]\s*$', line):
    return True
  elif re.match(r'^[\+\-]\s*\/\/*', line):
    return True
  else:
    return False

def detect_connected_minus_plus(line, minus_enter):
  ret = False
  if line.startswith('-'):
    minus_enter = True
  elif line.startswith('+') and minus_enter:
    ret = True
    minus_enter = False
  return ret, minus_enter
  
def delete_plus_minus(line):
  if line.startswith('+'):
    return line[1:]
  elif line.startswith('-'):
    return line[1:]
  else:
    return line
  
def add_del_line_cnt(line, line_add, line_del):
  if line.startswith('+'):
    line_add += 1
  elif line.startswith('-'):
    line_del += 1
  return line_add, line_del

def extract_class(line, summary, curr_class, braces_cnt, add_class_cnt, removed_class_cnt):
  class_pattern = r'^[\+\-]\s*(public|private|protected)?\s*class\s+(\w+[^{]*)'
  match =  re.match(class_pattern, line)
  if match:
    new_class =  "class "  + match.group(2).strip()
    if line.startswith('+'):
      if "new_class" not in summary:
        summary["new_class"] = {}
      summary["new_class"][new_class] = []
      add_class_cnt += 1
    elif line.startswith('-'):
      if "removed_class" not in summary:
        summary["removed_class"] = {}
      summary["removed_class"][new_class] = []
      removed_class_cnt += 1
    return new_class, summary, braces_cnt + 1, add_class_cnt, removed_class_cnt 
  else:
    return curr_class, summary, braces_cnt, add_class_cnt, removed_class_cnt


def extract_method(line, summary, curr_class, add_method_cnt, removed_method_cnt, add_func_cnt, removed_func_cnt):
  method_pattern = r'^[\+\-]\s*(?:(?:public|private|protected|static|final|native|synchronized|abstract|transient)+\s+)+[$\w<>\[\]\s]*\s+(\w+)\s*\('
  
  match =  re.match(method_pattern, line)
  curr_method = None
  
  if match:
    capture = match.start(1)
    line = line.strip()
    if line.endswith('{'):
      curr_method = str(line[capture:-1].strip())
    else:
      curr_method =  str(line[capture:].strip())
    
    if line.startswith('+'):
      if "new_class" not in summary:
        summary["new_class"] = {}
      
      if curr_class in summary["new_class"]:
        summary["new_class"][curr_class].append(curr_method)
        add_method_cnt += 1
      else:
        if "new_func" not in summary:
          summary["new_func"] = []
        summary["new_func"].append(curr_method) 
        add_func_cnt += 1
    
    elif line.startswith('-'):
      if "removed_class" not in summary:
        summary["removed_class"] = {}
      
      if curr_class in summary["removed_class"]:
        summary["removed_class"][curr_class].append(curr_method)
        removed_method_cnt += 1
      else:
        if "del_func" not in summary:
          summary["del_func"] = []
        summary["del_func"].append(curr_method)
        removed_func_cnt += 1
      
    return curr_method, summary, add_method_cnt, removed_method_cnt, add_func_cnt, removed_func_cnt
    
  else:
    return curr_method, summary, add_method_cnt, removed_method_cnt, add_func_cnt, removed_func_cnt

def extract_c_function(line, summary, add_func_cnt, removed_func_cnt):
  # function_pattern = r'^[\+\-]\s*(\w+\s+\*?\w+\s*\()'
  function_pattern = r'^[\+\-]\s*(?:const\s+)?(?:char\*|void|int|float|double|long|short|unsigned|signed)\s+\*?(\w+\s*\()'
  match = re.match(function_pattern, line)
  if match and not line.endswith(';'):
    capture = match.start(1)
    line = line.strip()
    if line.endswith('{'):
      curr_func = str(line[capture:-1].strip())
    else:
      curr_func =  str(line[capture:].strip())
      
    if line.startswith('+'):
      if "new_func" not in summary:
        summary["new_func"] = []
      summary["new_func"].append(curr_func)
      add_func_cnt += 1
    
    elif line.startswith('-'):
      if "del_func" not in summary:
        summary["del_func"] = []
      summary["del_func"].append(curr_func)
      removed_func_cnt += 1
    return summary, add_func_cnt, removed_func_cnt
  
  else: 
    return summary, add_func_cnt, removed_func_cnt

def count_code_import(line, include_cnt, ifndef_cnt, endif_cnt, import_cnt, define_cnt, ifdef_cnt, else_cnt):
  if re.match(r'^[\+\-]\s*#include', line):
    include_cnt += 1
  elif re.match(r'^[\+\-]\s*#ifndef', line):
    ifndef_cnt += 1
  elif re.match(r'^[\+\-]\s*#endif', line):
    endif_cnt += 1
  elif re.match(r'^[\+\-]\s*import', line):
    import_cnt += 1
  elif re.match(r'^[\+\-]\s*#define', line):
    define_cnt += 1
  elif re.match(r'^[\+\-]\s*#ifdef', line):
    ifdef_cnt += 1
  elif re.match(r'^[\+\-]\s*#else', line):
    else_cnt += 1
  return include_cnt, ifndef_cnt, endif_cnt, import_cnt, define_cnt, ifdef_cnt, else_cnt

def detect_variable(line):
  pass


def git_summary(diff):
  summary = {"diff_info": {}} 
  line_add = 0
  line_del = 0
  
  add_class_cnt = 0
  removed_class_cnt = 0
  add_method_cnt = 0
  removed_method_cnt = 0
  add_func_cnt = 0
  removed_func_cnt = 0
  
  include_cnt = 0
  ifndef_cnt = 0
  endif_cnt = 0
  import_cnt = 0
  define_cnt = 0
  ifdef_cnt = 0
  else_cnt = 0
  
  comment_cnt = 0
  minus_enter = False
  non_enter = False # bool for front is not neither + or -
  switch_cnt = 0
  
  bound_cnt = 3
  total_mod = 0
  large_mod = 0
  consecutive_plus = 0
  consecutive_minus = 0
  temp_sto_minus = 0
  temp_sto_plus = 0
  
  curr_class = None
  curr_method = None
  class_stack = []
  braces_cnt = 0
  
  lines = diff.split('\n')
  for line in lines:
    # set curr_class state to store class line and other variable
    curr_class, summary, braces_cnt, add_class_cnt, removed_class_cnt = extract_class(line, summary, curr_class, braces_cnt, add_class_cnt, removed_class_cnt)
    
    # store curr class(outer) in stack
    braces_cnt += line.count('{') - line.count('}')
    if(curr_class is not None):
      class_stack.append(curr_class)
      curr_class = class_stack[-1]
    
    # extract method
    curr_method, summary, add_method_cnt, removed_method_cnt, add_func_cnt, removed_func_cnt = extract_method(line, summary, curr_class, add_method_cnt, removed_method_cnt, add_func_cnt, removed_func_cnt)
    
    # extract function
    summary, add_func_cnt, removed_func_cnt = extract_c_function(line, summary, add_func_cnt, removed_func_cnt)
    
    if braces_cnt == 0 and class_stack:
      class_stack.pop()
    
    # count line add and line delete
    line_add, line_del = add_del_line_cnt(line, line_add, line_del)
    # count code module import and conditional compilation
    include_cnt, ifndef_cnt, endif_cnt, import_cnt, define_cnt, ifdef_cnt, else_cnt = count_code_import(line, include_cnt, ifndef_cnt, endif_cnt, import_cnt, define_cnt, ifdef_cnt, else_cnt)
    
    # comment cnt
    if(detect_comment_or_next_line(line)):
      comment_cnt += 1
      
    # count switch - to + -> possible for less important diff change
    
    if line.startswith('-'):
      consecutive_minus += 1
      consecutive_plus = 0
    elif line.startswith('+'):
      consecutive_plus += 1
      if consecutive_minus !=0 and consecutive_plus != 0:
        total_mod += 1
        temp_sto_minus = consecutive_minus
      consecutive_minus = 0
    else:
      temp_sto_plus = consecutive_plus
      if consecutive_plus != 0 and temp_sto_minus != 0:
        if max(temp_sto_minus, consecutive_plus) >= bound_cnt:
          large_mod += 1
      consecutive_plus = 0
      consecutive_minus = 0
      temp_sto_minus = 0
      temp_sto_plus = 0
  
  cond_cnt = ifndef_cnt + endif_cnt + define_cnt + ifdef_cnt + else_cnt
  import_include_cnt = import_cnt + include_cnt
  
  summary["diff_info"] = ({"la_n_ld": line_add + line_del,"la": line_add, "ld": line_del, "add_class": add_class_cnt, "del_class": removed_class_cnt,"add_method": add_method_cnt, "del_method": removed_method_cnt, "add_func": add_func_cnt, "del_func": removed_func_cnt, "cond_compilation": cond_cnt, "import_include": import_include_cnt, "comment": comment_cnt, "total_changed": total_mod, "few_changed": total_mod - large_mod, "many_changed": large_mod})
  
  
  natural_sum = json_to_natural_language(summary)
  
  return str(natural_sum)


