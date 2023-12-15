def json_to_natural_language(json_data):
    buffer = []

    diff_info = json_data.get('diff_info', {})
    if diff_info:
        diff_info_mod = (
            f"Total changed lines {diff_info.get('la_n_ld', 0)}, "
            f"with {diff_info.get('la', 0)} lines added and {diff_info.get('ld', 0)} lines removed. "
            f"Classes added: {diff_info.get('add_class', 0)}, classes removed: {diff_info.get('del_class', 0)}. "
            f"Methods added: {diff_info.get('add_method', 0)}, methods removed: {diff_info.get('del_method', 0)}. "
            f"Functions added: {diff_info.get('add_func', 0)}, functions removed: {diff_info.get('del_func', 0)}. "
            f"Conditional compilations: {diff_info.get('cond_compilation', 0)}, import/includes: {diff_info.get('import_include', 0)}. "
            f"Comments: {diff_info.get('comment', 0)}, few changes: {diff_info.get('few_changed', 0)}, "
            f"many changes: {diff_info.get('many_changed', 0)}."
        )
        buffer.append(diff_info_mod)

    for key in ['new_class', 'new_func', 'removed_class', 'del_func']:
        items = json_data.get(key, {})
        if items:
            if isinstance(items, dict):
                item_list = ', '.join(items.keys())
            else:
                item_list = ', '.join(items)
            buffer.append(f"{key.replace('_', ' ').capitalize()}: {item_list}.")

    return ' '.join(buffer)