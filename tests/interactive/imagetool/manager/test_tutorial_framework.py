from __future__ import annotations

import pytest
from qtpy import QtCore, QtGui, QtTest, QtWidgets

import erlab
import erlab.interactive.imagetool.manager._tutorial.framework as tutorial
from erlab.interactive._widgets import _CenteredIconToolButton


class _Emitter(QtCore.QObject):
    changed = QtCore.Signal()


def _shown_window(qtbot) -> QtWidgets.QWidget:
    window = QtWidgets.QWidget()
    window.resize(640, 480)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    return window


def test_widget_and_rect_target_geometry(qtbot) -> None:
    window = _shown_window(qtbot)
    button = QtWidgets.QPushButton(window)
    button.setGeometry(40, 50, 120, 30)
    button.show()

    geometry = tutorial.target_geometry(button)
    assert geometry is not None
    assert geometry.window is window
    assert geometry.rect.topLeft() == button.mapToGlobal(QtCore.QPoint())
    assert geometry.rect.size() == button.size()

    local_rect = QtCore.QRect(10, 15, 20, 25)
    geometry = tutorial.target_geometry(tutorial.RectTarget(local_rect, window))
    assert geometry is not None
    assert geometry.window is window
    assert geometry.rect.topLeft() == window.mapToGlobal(local_rect.topLeft())

    global_rect = QtCore.QRect(
        window.mapToGlobal(QtCore.QPoint(3, 4)), QtCore.QSize(8, 9)
    )
    geometry = tutorial.target_geometry(global_rect)
    assert geometry is not None
    assert geometry.rect == global_rect

    composite = tutorial.CompositeTarget(
        button, tutorial.RectTarget(local_rect, window)
    )
    geometry = tutorial.target_geometry(composite)
    assert geometry is not None
    assert geometry.rect == tutorial.target_geometry(button).rect.united(
        tutorial.target_geometry(tutorial.RectTarget(local_rect, window)).rect
    )


def test_action_target_geometry(qtbot) -> None:
    window = _shown_window(qtbot)
    menu = QtWidgets.QMenu(window)
    action = menu.addAction("Action")
    controller = tutorial.TourController(
        [
            tutorial.TourStep(
                "menu",
                "Menu",
                "Body",
                mode="action",
                target=tutorial.ActionTarget(action, menu),
            )
        ],
        window,
    )
    controller.start()
    qtbot.waitUntil(lambda: len(controller._overlays) == 1)
    menu.popup(window.mapToGlobal(QtCore.QPoint(20, 20)))
    qtbot.waitUntil(menu.isVisible)
    controller.notify_state_changed()

    geometry = tutorial.target_geometry(tutorial.ActionTarget(action, menu))
    assert geometry is not None
    expected = menu.actionGeometry(action)
    expected.moveTopLeft(menu.mapToGlobal(expected.topLeft()))
    assert geometry.rect == expected
    assert geometry.receivers == (menu,)
    assert controller._card is not None
    assert controller._card.isVisible()
    assert menu.isVisible()
    assert controller._overlays[0][0]() is window
    assert controller._overlays[0][1]._spotlight is None
    controller.close()
    menu.close()


def test_model_row_and_model_index_target_geometry(qtbot) -> None:
    window = _shown_window(qtbot)
    view = QtWidgets.QListView(window)
    view.setGeometry(20, 20, 220, 180)
    model = QtGui.QStandardItemModel(view)
    model.appendRow(QtGui.QStandardItem("First"))
    model.appendRow(QtGui.QStandardItem("Second"))
    view.setModel(model)
    view.show()
    qtbot.waitUntil(lambda: not view.visualRect(model.index(1, 0)).isEmpty())

    row_geometry = tutorial.target_geometry(tutorial.ModelRowTarget(view, 1))
    index_geometry = tutorial.target_geometry(model.index(1, 0))
    assert row_geometry is not None
    assert index_geometry is not None
    assert row_geometry.rect == index_geometry.rect
    assert row_geometry.receivers == (view, view.viewport())


def test_graphics_item_target_geometry(qtbot) -> None:
    window = _shown_window(qtbot)
    view = QtWidgets.QGraphicsView(window)
    view.setGeometry(20, 20, 300, 240)
    scene = QtWidgets.QGraphicsScene(view)
    item = scene.addRect(QtCore.QRectF(10, 15, 40, 30))
    view.setScene(scene)
    view.show()

    geometry = tutorial.target_geometry(tutorial.GraphicsItemTarget(item, view))
    assert geometry is not None
    expected = view.mapFromScene(item.sceneBoundingRect()).boundingRect()
    expected.moveTopLeft(view.viewport().mapToGlobal(expected.topLeft()))
    assert geometry.rect == expected
    assert geometry.window is window


def test_information_step_readiness_and_transition(qtbot) -> None:
    window = _shown_window(qtbot)
    ready = False
    steps = [
        tutorial.TourStep(
            "welcome",
            "Welcome",
            "Body",
            target_required=False,
            ready=lambda: ready,
            continue_label="Start",
        ),
        tutorial.TourStep(
            "finish", "Finish", "Body", target_required=False, continue_label="Finish"
        ),
    ]
    controller = tutorial.TourController(steps, window)
    controller.start()
    qtbot.waitUntil(lambda: bool(controller._overlays))
    card = controller._card
    assert card is not None
    assert not card.skip_button.isVisible()
    assert not card.continue_button.isEnabled()
    qtbot.mouseClick(card.continue_button, QtCore.Qt.MouseButton.LeftButton)
    assert controller.current_step is steps[0]

    ready = True
    controller.update_current(title="Ready", body="Ready body")
    controller.notify_state_changed()
    assert card.continue_button.isEnabled()
    qtbot.mouseClick(card.continue_button, QtCore.Qt.MouseButton.LeftButton)
    assert controller.current_step is steps[1]
    with qtbot.waitSignal(controller.finished):
        qtbot.mouseClick(card.continue_button, QtCore.Qt.MouseButton.LeftButton)
    assert not controller.is_running


def test_debug_skip_advances_information_and_action_steps(qtbot) -> None:
    window = _shown_window(qtbot)
    action_complete = False

    def perform_action() -> None:
        nonlocal action_complete
        action_complete = True

    steps = [
        tutorial.TourStep("overview", "Overview", "Body", target_required=False),
        tutorial.TourStep(
            "action",
            "Action",
            "Body",
            mode="action",
            target_required=False,
            completion=lambda: action_complete,
            debug_action=perform_action,
            auto_advance=False,
        ),
        tutorial.TourStep("finish", "Finish", "Body", target_required=False),
    ]
    controller = tutorial.TourController(steps, window, debug=True)
    controller.start()
    card = controller._card
    assert card is not None
    assert card.skip_button.isVisible()

    qtbot.mouseClick(card.skip_button, QtCore.Qt.MouseButton.LeftButton)
    assert controller.current_step is steps[1]
    qtbot.mouseClick(card.skip_button, QtCore.Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: controller.current_step is steps[2])
    assert action_complete
    controller.close()


def test_debug_skip_requires_an_action(qtbot) -> None:
    window = _shown_window(qtbot)
    step = tutorial.TourStep(
        "action",
        "Action",
        "Body",
        mode="action",
        target_required=False,
        completion=lambda: False,
    )
    controller = tutorial.TourController([step], window, debug=True)
    controller.start()
    with pytest.raises(tutorial.TutorialDebugActionError, match="'action'"):
        controller._card_skip()
    controller.close()


def test_ui_text_placeholders_resolve_lazily(qtbot) -> None:
    window = _shown_window(qtbot)
    labels = {
        "action": "&Open && Inspect",
        "control": "Preview",
        "continue": "Proceed",
    }
    controller = tutorial.TourController(
        [
            tutorial.TourStep(
                "labels",
                "Use [[ui:action]]",
                "Select [[ui:control]].",
                target_required=False,
                hint="The [[ui:control]] control is available.",
                continue_label="[[ui:continue]]",
            )
        ],
        window,
        text_resolver=labels.get,
    )
    controller.start()
    card = controller._card
    assert card is not None
    assert card.title.text() == "Use &Open && Inspect"
    assert card.body.text() == "Select Preview."
    assert card.hint.text() == "The Preview control is available."
    assert card.continue_button.text() == "Proceed"

    labels["control"] = "Data Preview"
    controller.notify_state_changed()
    assert card.body.text() == "Select Data Preview."
    assert card.hint.text() == "The Data Preview control is available."
    controller.close()


def test_menu_path_uses_semantic_inline_segments(qtbot) -> None:
    window = _shown_window(qtbot)
    labels = {
        "menu": "View",
        "submenu": "Rotation Guidelines",
        "action": "C6",
    }
    step = tutorial.TourStep(
        "menu-path",
        "Use the menu",
        "Select [[menu:menu|submenu|action]].",
        target_required=False,
    )
    controller = tutorial.TourController([step], window, text_resolver=labels.get)
    controller.start()
    card = controller._card
    assert card is not None
    assert card.body.text() == (
        "Select View \N{SINGLE RIGHT-POINTING ANGLE QUOTATION MARK} Rotation "
        "Guidelines \N{SINGLE RIGHT-POINTING ANGLE QUOTATION MARK} C6."
    )
    assert [span.kind for span in card.body._text.spans] == [
        "plain",
        "menu",
        "menu_separator",
        "menu",
        "menu_separator",
        "menu_action",
        "plain",
    ]
    assert card.body.accessibleName() == card.body.text()
    controller.close()


def test_tutorial_colors_use_palette_roles(qtbot) -> None:
    window = _shown_window(qtbot)
    controller = tutorial.TourController(
        [tutorial.TourStep("step", "Step", "Body", target_required=False)],
        window,
    )
    controller.start()
    card = controller._card
    assert card is not None
    roles = (
        tutorial._CARD_BACKGROUND_ROLE,
        tutorial._CARD_BORDER_ROLE,
        tutorial._TEXT_ROLE,
        tutorial._MUTED_TEXT_ROLE,
        tutorial._UI_BACKGROUND_ROLE,
        tutorial._UI_BORDER_ROLE,
        tutorial._UI_TEXT_ROLE,
        tutorial._MENU_BACKGROUND_ROLE,
        tutorial._MENU_BORDER_ROLE,
        tutorial._MENU_ACTION_BORDER_ROLE,
        tutorial._MENU_TEXT_ROLE,
        tutorial._BUTTON_TEXT_ROLE,
        tutorial._OVERLAY_ROLE,
        tutorial._SPOTLIGHT_BORDER_ROLE,
    )
    assert all(isinstance(role, QtGui.QPalette.ColorRole) for role in roles)
    assert card.backgroundRole() == QtGui.QPalette.ColorRole.Base
    assert card.foregroundRole() == QtGui.QPalette.ColorRole.Text
    assert card.progress.foregroundRole() == QtGui.QPalette.ColorRole.PlaceholderText
    controller.close()


def test_missing_ui_text_waits_without_retry(qtbot) -> None:
    window = _shown_window(qtbot)
    labels: dict[str, str] = {}
    controller = tutorial.TourController(
        [
            tutorial.TourStep(
                "labels",
                "Use the control",
                "Select [[ui:control]].",
                target_required=False,
            )
        ],
        window,
        text_resolver=labels.get,
    )
    controller.start()
    card = controller._card
    assert card is not None
    assert card.continue_button.text() == "Next"
    assert not card.continue_button.isEnabled()

    labels["control"] = "Preview"
    controller.notify_state_changed()
    assert card.title.text() == "Use the control"
    assert card.body.text() == "Select Preview."
    assert card.continue_button.text() == "Next"
    assert card.continue_button.isEnabled()
    controller.close()


def test_malformed_ui_text_placeholder_is_unavailable(qtbot) -> None:
    window = _shown_window(qtbot)
    controller = tutorial.TourController(
        [
            tutorial.TourStep(
                "labels",
                "Use the control",
                "Select [[ui:control].",
                target_required=False,
                timeout_ms=0,
            )
        ],
        window,
        text_resolver=lambda _object_name: "Control",
    )
    with pytest.raises(tutorial.TutorialStepUnavailableError, match="'labels'"):
        controller.start()
    controller.close()


def test_malformed_menu_path_placeholder_is_unavailable(qtbot) -> None:
    window = _shown_window(qtbot)
    controller = tutorial.TourController(
        [
            tutorial.TourStep(
                "labels",
                "Use the menu",
                "Select [[menu:menu]].",
                target_required=False,
                timeout_ms=0,
            )
        ],
        window,
        text_resolver=lambda _object_name: "Menu",
    )
    with pytest.raises(tutorial.TutorialStepUnavailableError, match="'labels'"):
        controller.start()
    controller.close()


def test_instruction_card_is_not_a_window(qtbot) -> None:
    window = _shown_window(qtbot)
    button = QtWidgets.QPushButton(window)
    button.setGeometry(40, 50, 120, 30)
    button.show()
    controller = tutorial.TourController(
        [tutorial.TourStep("overview", "Overview", "Body", target=button)],
        window,
    )
    controller.start()
    qtbot.waitUntil(lambda: bool(controller._overlays))

    overlay = controller._overlays[0][1]
    card = controller._card
    assert isinstance(card, QtWidgets.QFrame)
    assert not card.isWindow()
    assert card.parentWidget() is window
    assert card.isVisible()
    assert overlay._spotlight is not None
    assert not controller.eventFilter(
        card.continue_button, QtCore.QEvent(QtCore.QEvent.Type.MouseButtonPress)
    )
    assert window.rect().contains(card.geometry())

    clicks: list[bool] = []
    card.continue_button.clicked.connect(lambda: clicks.append(True))
    window_handle = window.windowHandle()
    assert window_handle is not None
    button_center = card.continue_button.mapTo(
        window, card.continue_button.rect().center()
    )
    hit = QtWidgets.QApplication.widgetAt(
        card.continue_button.mapToGlobal(card.continue_button.rect().center())
    )
    assert controller._object_contains(card.continue_button, hit)
    QtTest.QTest.mouseClick(
        window_handle,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        button_center,
    )
    assert clicks == [True]

    controller.close()


def test_instruction_card_can_be_centered(qtbot) -> None:
    window = _shown_window(qtbot)
    target = QtWidgets.QPushButton(window)
    target.setGeometry(8, 8, 80, 30)
    target.show()
    controller = tutorial.TourController(
        [
            tutorial.TourStep(
                "switch",
                "Switch windows",
                "Body",
                target=target,
                card_position="center",
            )
        ],
        window,
    )
    controller.start()
    qtbot.waitUntil(lambda: bool(controller._overlays))

    card = controller._card
    assert card is not None
    offset = card.geometry().center() - window.rect().center()
    assert offset.manhattanLength() <= 2
    controller.close()


def test_instruction_card_recalculates_height_for_narrow_window(qtbot) -> None:
    window = _shown_window(qtbot)
    window.resize(260, 420)
    controller = tutorial.TourController(
        [
            tutorial.TourStep(
                "narrow",
                "Narrow tutorial card",
                "Select [[ui:control]] to apply the coordinate offset. The second "
                "sentence must wrap without being cut off.",
                target_required=False,
            )
        ],
        window,
        text_resolver=lambda _object_name: "Offset",
    )
    controller.start()
    card = controller._card
    assert card is not None
    qtbot.waitUntil(card.isVisible)

    assert window.rect().contains(card.geometry())
    assert card.body.height() >= card.body.heightForWidth(card.body.width())
    layout = card.layout()
    assert layout is not None
    assert card.height() >= layout.heightForWidth(card.width())
    controller.close()


def test_instruction_card_header_drag_preserves_position_during_refresh(qtbot) -> None:
    window = _shown_window(qtbot)
    target = QtWidgets.QPushButton(window)
    target.setGeometry(8, 8, 80, 30)
    target.show()
    controller = tutorial.TourController(
        [
            tutorial.TourStep("first", "First", "Body", target=target),
            tutorial.TourStep("second", "Second", "Body", target=target),
        ],
        window,
    )
    controller.start()
    card = controller._card
    assert card is not None
    qtbot.waitUntil(card.isVisible)

    initial = card.pos()
    press_position = card.header.rect().center()
    drag_delta = QtCore.QPoint(60, 8)
    qtbot.mousePress(
        card.header,
        QtCore.Qt.MouseButton.LeftButton,
        pos=press_position,
    )
    qtbot.mouseMove(card.header, pos=press_position + drag_delta)
    qtbot.mouseRelease(
        card.header,
        QtCore.Qt.MouseButton.LeftButton,
        pos=press_position + drag_delta,
    )

    moved = card.pos()
    assert moved != initial
    assert window.rect().contains(card.geometry())
    controller.notify_state_changed()
    assert card.pos() == moved

    controller.continue_step()
    assert controller.current_step is not None
    assert controller.current_step.id == "second"
    assert card._manual_position is None
    controller.close()


def test_instruction_card_survives_target_window_close(qtbot) -> None:
    window = _shown_window(qtbot)
    dialog = QtWidgets.QDialog(window)
    dialog.setWindowFlag(QtCore.Qt.WindowType.Window, True)
    dialog.resize(320, 240)
    dialog.show()
    qtbot.waitExposed(dialog)
    controller = tutorial.TourController(
        [tutorial.TourStep("dialog", "Dialog", "Body", target=dialog)],
        window,
    )
    controller.start()
    qtbot.waitUntil(lambda: controller._card is not None)
    card = controller._card
    assert card is not None
    qtbot.waitUntil(lambda: card.parentWidget() is dialog)

    dialog.close()
    qtbot.waitUntil(lambda: card.parentWidget() is window)
    assert erlab.interactive.utils.qt_is_valid(card, card.continue_button)
    controller.close()


def test_transient_popup_does_not_own_instruction_card(qtbot) -> None:
    window = _shown_window(qtbot)
    dialog = QtWidgets.QDialog(window)
    dialog.setWindowFlag(QtCore.Qt.WindowType.Window, True)
    dialog.resize(480, 320)
    target = QtWidgets.QPushButton("Target", dialog)
    target.move(20, 20)
    dialog.show()
    qtbot.waitExposed(dialog)
    complete = False
    controller = tutorial.TourController(
        [
            tutorial.TourStep(
                "dialog-action",
                "Dialog action",
                "Body",
                mode="action",
                target=target,
                completion=lambda: complete,
                auto_advance=False,
            )
        ],
        window,
    )
    controller.start()
    card = controller._card
    assert card is not None
    qtbot.waitUntil(lambda: card.parentWidget() is dialog)

    popup = QtWidgets.QWidget(dialog, QtCore.Qt.WindowType.Popup)
    popup.resize(120, 80)
    popup.show()
    qtbot.waitUntil(popup.isVisible)
    popup.activateWindow()
    qtbot.waitUntil(lambda: QtWidgets.QApplication.activeWindow() is popup)
    complete = True
    controller.notify_state_changed()

    assert popup not in controller._visible_windows()
    assert card.parentWidget() is dialog
    popup.hide()
    QtWidgets.QApplication.processEvents()
    assert card.isVisible()
    controller.close()


def test_overlay_darkens_untargeted_area(qtbot) -> None:
    window = _shown_window(qtbot)
    window.setStyleSheet("background-color: white")
    button = QtWidgets.QPushButton("Target", window)
    button.setGeometry(40, 50, 120, 30)
    button.show()
    QtWidgets.QApplication.processEvents()
    before = window.grab().toImage()

    controller = tutorial.TourController(
        [tutorial.TourStep("spotlight", "Spotlight", "Body", target=button)],
        window,
    )
    controller.start()
    qtbot.waitUntil(lambda: bool(controller._overlays))
    QtWidgets.QApplication.processEvents()
    after = window.grab().toImage()

    outside = QtCore.QPoint(300, 300)
    target = button.geometry().center()
    rounded_corner = button.geometry().topLeft() - QtCore.QPoint(5, 5)
    assert (
        after.pixelColor(outside).lightness() < before.pixelColor(outside).lightness()
    )
    assert (
        after.pixelColor(outside).lightness()
        > before.pixelColor(outside).lightness() / 2
    )
    assert after.pixelColor(target) == before.pixelColor(target)
    assert (
        after.pixelColor(rounded_corner).lightness()
        < before.pixelColor(rounded_corner).lightness()
    )
    controller.close()


def test_action_step_requires_continue_after_observed_state(qtbot) -> None:
    window = _shown_window(qtbot)
    emitter = _Emitter()
    complete = False
    steps = [
        tutorial.TourStep(
            "action",
            "Action",
            "Body",
            mode="action",
            target_required=False,
            subscriptions=(lambda: emitter.changed,),
            completion=lambda: complete,
            auto_advance=False,
        ),
        tutorial.TourStep("done", "Done", "Body", target_required=False),
    ]
    controller = tutorial.TourController(steps, window)
    controller.start()
    card = controller._card
    assert card is not None
    assert card.continue_button.isVisible()
    assert not card.continue_button.isEnabled()
    emitter.changed.emit()
    assert controller.current_step is steps[0]
    complete = True
    emitter.changed.emit()
    assert controller.current_step is steps[0]
    assert card.continue_button.isEnabled()
    qtbot.mouseClick(card.continue_button, QtCore.Qt.MouseButton.LeftButton)
    assert controller.current_step is steps[1]

    controller.close()
    emitter.changed.emit()
    assert controller.current_step is None


def test_discrete_action_advances_on_observed_state(qtbot) -> None:
    window = _shown_window(qtbot)
    emitter = _Emitter()
    complete = False
    steps = [
        tutorial.TourStep(
            "action",
            "Action",
            "Body",
            mode="action",
            target_required=False,
            subscriptions=(lambda: emitter.changed,),
            completion=lambda: complete,
        ),
        tutorial.TourStep("done", "Done", "Body", target_required=False),
    ]
    controller = tutorial.TourController(steps, window)
    controller.start()
    complete = True
    emitter.changed.emit()
    assert controller.current_step is steps[1]
    controller.close()


def test_input_gating_and_escape(qtbot) -> None:
    window = _shown_window(qtbot)
    target = QtWidgets.QPushButton(window)
    target.setGeometry(20, 20, 100, 30)
    target.show()
    outside = QtWidgets.QPushButton(window)
    outside.setGeometry(20, 80, 100, 30)
    outside.show()
    step = tutorial.TourStep(
        "action", "Action", "Body", mode="action", target=lambda: target
    )
    controller = tutorial.TourController([step], window)
    controller.start()
    controller.notify_state_changed()
    card = controller._card
    assert card is not None
    assert isinstance(card.exit_button, _CenteredIconToolButton)
    assert card.exit_button.parentWidget() is card.header
    assert card.exit_button.autoRaise()
    assert not card.exit_button.icon().isNull()
    assert card.exit_button.accessibleName()
    overlay = controller._overlays[0][1]
    assert overlay.testAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    clicks: list[bool] = []
    target.clicked.connect(lambda: clicks.append(True))
    window_handle = window.windowHandle()
    assert window_handle is not None
    QtTest.QTest.mouseClick(
        window_handle,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        target.mapTo(window, target.rect().center()),
    )
    assert clicks == [True]

    outside_clicks: list[bool] = []
    outside.clicked.connect(lambda: outside_clicks.append(True))
    QtTest.QTest.mouseClick(
        window_handle,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        outside.mapTo(window, outside.rect().center()),
    )
    assert outside_clicks == []

    mouse = QtCore.QEvent(QtCore.QEvent.Type.MouseButtonPress)
    assert not controller.eventFilter(target, mouse)
    assert controller.eventFilter(outside, QtCore.QEvent(QtCore.QEvent.Type.Wheel))
    assert controller.eventFilter(outside, QtCore.QEvent(QtCore.QEvent.Type.TouchBegin))
    assert controller.eventFilter(outside, QtCore.QEvent(QtCore.QEvent.Type.DragEnter))
    assert controller.eventFilter(outside, QtCore.QEvent(QtCore.QEvent.Type.Drop))
    assert controller.eventFilter(
        outside, QtCore.QEvent(QtCore.QEvent.Type.ContextMenu)
    )

    exits: list[bool] = []
    controller.exit_requested.connect(lambda: exits.append(True))
    qtbot.mouseClick(card.exit_button, QtCore.Qt.MouseButton.LeftButton)
    assert exits == [True]
    exits.clear()
    escape = QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress,
        QtCore.Qt.Key.Key_Escape,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    assert controller.eventFilter(outside, escape)
    assert exits == [True]
    assert controller.is_running
    controller.close()


def test_allowed_objects_and_event_predicate(qtbot) -> None:
    window = _shown_window(qtbot)
    allowed = QtWidgets.QLineEdit(window)
    allowed.show()
    other = QtWidgets.QLineEdit(window)
    other.show()
    step = tutorial.TourStep(
        "info",
        "Info",
        "Body",
        target_required=False,
        allowed_objects=(lambda: allowed,),
        event_predicate=lambda watched, event: (
            watched is other and event.type() == QtCore.QEvent.Type.ContextMenu
        ),
    )
    controller = tutorial.TourController([step], window)
    controller.start()

    assert not controller.eventFilter(
        allowed, QtCore.QEvent(QtCore.QEvent.Type.KeyPress)
    )
    assert not controller.eventFilter(
        other, QtCore.QEvent(QtCore.QEvent.Type.ContextMenu)
    )
    assert controller.eventFilter(other, QtCore.QEvent(QtCore.QEvent.Type.KeyPress))
    controller.close()


def test_message_dialog_details_toggle_bypasses_input_gating(qtbot) -> None:
    window = _shown_window(qtbot)
    target = QtWidgets.QPushButton(window)
    target.show()
    controller = tutorial.TourController(
        [
            tutorial.TourStep(
                "action",
                "Action",
                "Body",
                mode="action",
                target=target,
            )
        ],
        window,
    )
    controller.start()
    dialog = erlab.interactive.utils.MessageDialog(
        parent=window,
        title="Error",
        text="An error occurred",
        detailed_text="Traceback",
    )
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(dialog._details_toggle.isVisible)

    assert not controller.eventFilter(
        dialog._details_toggle,
        QtCore.QEvent(QtCore.QEvent.Type.MouseButtonPress),
    )
    qtbot.mouseClick(dialog._details_toggle, QtCore.Qt.MouseButton.LeftButton)
    assert dialog._details_container.isVisible()

    controller.close()


def test_transient_missing_target_waits_without_retry(qtbot) -> None:
    window = _shown_window(qtbot)
    target: QtWidgets.QWidget | None = None
    reveals: list[bool] = []
    step = tutorial.TourStep(
        "missing",
        "Missing",
        "Body",
        target=lambda: target,
        reveal=lambda: reveals.append(True),
        timeout_ms=1000,
        retry_interval_ms=5,
    )
    controller = tutorial.TourController([step], window)
    controller.start()
    qtbot.waitUntil(lambda: bool(controller._overlays))
    card = controller._card
    assert card is not None
    assert card.continue_button.isVisible()
    assert card.continue_button.text() == "Next"
    assert not card.continue_button.isEnabled()
    assert reveals == [True]

    target = QtWidgets.QPushButton(window)
    target.show()
    controller.notify_state_changed()
    assert controller.is_running
    assert card.continue_button.isEnabled()
    target.deleteLater()
    qtbot.waitUntil(lambda: not erlab.interactive.utils.qt_is_valid(target))
    controller.notify_state_changed()
    assert controller.is_running
    assert not card.continue_button.isEnabled()
    controller.close()


def test_missing_target_raises_diagnostic_error(qtbot) -> None:
    window = _shown_window(qtbot)
    controller = tutorial.TourController(
        [
            tutorial.TourStep(
                "missing-target",
                "Missing target",
                "Body",
                target=lambda: None,
                timeout_ms=0,
            )
        ],
        window,
    )
    with pytest.raises(
        tutorial.TutorialStepUnavailableError, match="'missing-target'"
    ) as error:
        controller.start()
    assert controller._fatal_error is error.value
    assert not controller._retry_timer.isActive()
    controller.notify_state_changed()
    controller.close()


def test_overlay_resize_cleanup(qtbot) -> None:
    window = _shown_window(qtbot)
    step = tutorial.TourStep("step", "Step", "Body", target_required=False)
    controller = tutorial.TourController([step], window)
    controller.start()
    qtbot.waitUntil(lambda: bool(controller._overlays))
    overlay = controller._overlays[0][1]
    card = controller._card
    assert card is not None

    window.resize(720, 520)
    qtbot.waitUntil(lambda: overlay.size() == window.size())
    assert window.rect().contains(card.geometry())

    controller.close()
    assert controller._overlays == []
    assert controller._card is None
    assert controller._connections == []
    assert not controller._retry_timer.isActive()
