def add_to_db(inf, all_id):
    from track import conn, cursor
    add = []
    for i in all_id:
        inf['accuracy%s' % i] = round(inf['accuracy%s' % i] / (inf['time%s' % i]), 2)
        number = str(i)
        name = str(inf['name%s' % i])
        accuracy = str(inf['accuracy%s' % i])
        time_ent = str(inf['time_ent_%s' % i])
        time_out = str(inf['time_out_%s' % i])
        add.append([number, name, time_ent, time_out, accuracy])

        try:
            cursor.execute('INSERT INTO main.info (number, fight_fall, time_ent, time_out, average_accuracy) '
                              'VALUES (%s, %s, %s, %s, %s);', (number, name, time_ent, time_out, accuracy))
            conn.commit()
        except:
            continue

    return add
