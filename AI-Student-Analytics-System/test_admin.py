import sys, os
sys.path.insert(0, '.')
from app import app, init_db
init_db()
with app.test_client() as c:
    c.post('/login', data={'username':'admin','password':'admin123'}, follow_redirects=True)
    r = c.get('/admin', follow_redirects=True)
    print('Admin status:', r.status_code)
    html = r.data.decode('utf-8', errors='replace')
    if 'TemplateSyntaxError' in html or 'TemplateAssertionError' in html:
        print('TEMPLATE ERROR:', html[:600])
    else:
        print('Admin page OK')
        for word in ['Admin Panel', 'User Management', 'Total Predictions', 'System Information']:
            print(f'  Contains "{word}":', word in html)
print('Done.')
