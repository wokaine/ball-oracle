# Textual
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll, HorizontalScroll
from textual.widgets import Header, Static, Footer, OptionList, Placeholder, Checkbox, Button, Input, LoadingIndicator
from textual.screen import Screen
from textual_pandas.widgets import DataFrameTable
from textual.widgets.option_list import Option

# Generic imports
import random
import constants
import os
import pandas as pd

# Custom
from hiddengemfinder.hidden_gem_finder import HiddenGemFinder
from comebackcalc.comeback_calculator import ComebackCalculator

############################
### MAIN MENU COMPONENTS ###
############################

class AppHeader(Static):
    def compose(self) -> ComposeResult:
        random_quote = random.choice(constants.QUOTES)
        yield Static(constants.MOTD_APP, id="main-ascii-art", expand=True)
        yield Static(f"[i]{random_quote[0]}[/i] - {random_quote[1]}")

class AboutPanel(Static):
    def compose(self) -> ComposeResult:
        yield Static(constants.ABOUT_MAIN, id="main-about-text")

class ProgramMenu(Vertical):
    def compose(self) -> ComposeResult:
        yield Static("[reverse] PROGRAM SELECT [/]")
        yield OptionList(
            Option("Hidden Gem Finder", id="hgf"),
            Option("Comeback Calculator", id="cbc"),
            Option("Market Value Predictor (WIP)", id="mvp"),
            id="main-menu-list"
        )

class MainMenu(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-top-section", classes="top"):
            yield AppHeader(classes="header")
            yield AboutPanel(classes="about")
        yield ProgramMenu()
        yield Footer()

####################################
### HIDDEN GEM FINDER COMPONENTS ###
####################################
        
class HGFHeader(Static):
    def compose(self) -> ComposeResult:
        yield Static(constants.MOTD_HGF, id="hgf-ascii-art", expand=True)

class HGFAbout(Static):
    def compose(self) -> ComposeResult:
        yield Static(constants.ABOUT_HGF, id="hgf-about")

class HGFLeft(Static):
    def compose(self) -> ComposeResult:
        with Vertical():
            with VerticalScroll(id="hgf-vscrollbox"):
                yield Checkbox("Premier League", id="EPL")
                yield Checkbox("La Liga", id="La_Liga")
                yield Checkbox("Bundesliga", id="Bundesliga")
                yield Checkbox("Serie A", id="Serie_A")
                yield Checkbox("Ligue 1", id="Ligue_1")
                yield Checkbox("Russian Premier League", id="RFPL")
            yield Input(placeholder="Search for a player", type="text")
            yield Button("Submit", id="hgf-submit")

class HGFBody(Static):
    def compose(self) -> ComposeResult:
        with Horizontal():
            yield HGFLeft()
            with HorizontalScroll(id="hgf-hscrollbox"):
                yield DataFrameTable()
    
    def on_mount(self) -> None:
        self.hgf = HiddenGemFinder()

        checkbox = self.query_one(VerticalScroll)
        checkbox.border_title = "League"

        dftable = self.query_one(HorizontalScroll)
        dftable.border_title = "Results"
    
    def on_button_pressed(self) -> None:
        leagues = [cb.id for cb in self.query(Checkbox) if cb.value]
        player = self.query_one(Input).value
        if len(leagues) > 0:
            df = self.execute_hgf(leagues=leagues, player=player)
            dftable = self.query_one(DataFrameTable)
            dftable.update_df(df)

    def execute_hgf(self, leagues, player=''):
        self.hgf.update_league(leagues=leagues)
        data_all = self.hgf.data
        if player == '':
            return data_all
        else:
            self.hgf.update_player(player=player)
            similar_players = self.hgf.get_query()
            if similar_players is not None:  
                # i.e. player found       
                return similar_players
            return data_all
        

class HiddenGemFinderScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="hgf-top-section", classes="top"):
            yield HGFHeader(classes="header")
            yield HGFAbout(classes="about")
        yield HGFBody()
        yield Footer()

######################################
### COMEBACK CALCULATOR COMPONENTS ###
######################################

class CCHeader(Static):
    def compose(self) -> ComposeResult:
        yield Static(constants.MOTD_CC, id="cc-ascii-art", expand=True)

class CCAbout(Static):
    def compose(self) -> ComposeResult:
        yield Static(constants.ABOUT_CC, id="cc-about")

class CCBody(Static):
    def __init__(self) -> None:
        super().__init__()
        xgb_file = os.path.abspath("comeback_calc.json")
        calib_file = os.path.abspath("calibrated_model.pkl")
        self.cc = ComebackCalculator(xgb_model_file=xgb_file, calib_model_file=calib_file)

        # Data from the 25/26 season thus far
        tfts = pd.read_csv("https://www.football-data.co.uk/mmz4281/2526/E0.csv")
        self.matches = self.cc.predict_season(tfts)

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="cc-vscroll"):
            for _, m in self.matches.iterrows():
                yield CCMatch(match=m)
        
class CCMatch(Static):
    def __init__(self, match):
        super().__init__()
        self.match = match
 
    def compose(self) -> ComposeResult:
        yield Static(f"==== {self.match['Date']} | {self.match['Time']} ====")
        yield Static(f"{self.match['HomeTeam']} vs. {self.match['AwayTeam']}")
        yield Static(f"HALF-TIME RESULT: {self.match['HTHG']} - {self.match['HTAG']}")
        yield Static(f"FULL-TIME RESULT: {self.match['FTHG']} - {self.match['FTAG']}")
        format_prediction = "Comeback" if self.match['Model_Prediction'] else "No Comeback"
        yield Static(f"MODEL PREDICTON: {format_prediction} with probability {self.match['Comeback_Probability']:.4f}")
        format_outcome = "Comeback" if self.match['Actual_Outcome'] else "No Comeback"
        yield Static(f"ACTUAL OUTCOME: {format_outcome}\n")

class ComebackCalculatorScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="cc-top-section", classes="top"):
            yield CCHeader(classes="header")
            yield CCAbout(classes="about")
        yield CCBody()
        yield Footer()

#########################################
### MARKET VALUE PREDICTOR COMPONENTS ###
#########################################

class MarketValuePredictor(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Placeholder("Market Value Predictor")
        yield Footer()

class BallOracle(App):
    """Main UI for the Ball Oracle program"""

    CSS_PATH = "./style/main.tcss"

    MODES = {
        "main menu": MainMenu,
        "hgf": HiddenGemFinderScreen,
        "cbc": ComebackCalculatorScreen,
        "mvp": MarketValuePredictor
    }

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("b", "switch_mode('main menu')", "Back")
    ]

    # Default
    def on_mount(self) -> None:
        self.switch_mode("main menu")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected):
        self.switch_mode(event.option_id)

if __name__ == "__main__":
    app = BallOracle()
    app.run()