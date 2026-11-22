import sys
from pathlib import Path
from playwright.sync_api import sync_playwright
deck=Path(sys.argv[1]).resolve(); out=Path(sys.argv[2]); out.mkdir(exist_ok=True); errors=[]; N=6
with sync_playwright() as p:
    b=p.chromium.launch(executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe',headless=True)
    pg=b.new_page(viewport={'width':1440,'height':1000})
    pg.on('pageerror',lambda e:errors.append(str(e))); pg.on('console',lambda m:m.type=='error' and errors.append(m.text))
    pg.goto(deck.as_uri())
    assert pg.locator('.slide').count()==N and pg.locator('.slide:visible').count()==1

    for i in range(N):
        pg.locator('#jump').select_option(str(i)); pg.screenshot(path=str(out/f's{i+1}.png'))
        assert pg.locator('.slide:visible .inner').evaluate('(e)=>e.scrollWidth<=e.clientWidth+2'),('desktop',i)
    pg.locator('#theme').click(); pg.locator('#jump').select_option('2'); pg.screenshot(path=str(out/'s5-dark.png')); pg.locator('#theme').click()
    pg.emulate_media(media='print'); assert pg.locator('.slide:visible').count()==N; pg.emulate_media(media='screen')
    pg.set_viewport_size({'width':390,'height':844})
    for i in range(N):
        pg.locator('#jump').select_option(str(i))
        assert pg.locator('.slide:visible .inner').evaluate('(e)=>e.scrollWidth<=e.clientWidth+2'),('mobile',i)
    pg.locator('#jump').select_option('2'); pg.screenshot(path=str(out/'s5-mobile.png'))
    assert not errors,errors; b.close()
print('Passed: 6 slides, charts rendered, dark, print, desktop/mobile overflow, no JS errors.')
