/*******************************************************************************
 * Copyright 2016-2017 Dell Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 * @microservice: core-metadata
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.controller;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.mockito.Matchers.any;
import static org.mockito.Mockito.when;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

import org.edgexfoundry.controller.impl.CallbackExecutor;
import org.edgexfoundry.controller.impl.ScheduleEventControllerImpl;
import org.edgexfoundry.dao.AddressableDao;
import org.edgexfoundry.dao.ScheduleEventDao;
import org.edgexfoundry.dao.ScheduleEventRepository;
import org.edgexfoundry.dao.ScheduleRepository;
import org.edgexfoundry.domain.meta.Addressable;
import org.edgexfoundry.domain.meta.ScheduleEvent;
import org.edgexfoundry.exception.controller.DataValidationException;
import org.edgexfoundry.exception.controller.LimitExceededException;
import org.edgexfoundry.exception.controller.NotFoundException;
import org.edgexfoundry.exception.controller.ServiceException;
import org.edgexfoundry.test.category.RequiresNone;
import org.edgexfoundry.test.data.AddressableData;
import org.edgexfoundry.test.data.ScheduleData;
import org.edgexfoundry.test.data.ScheduleEventData;
import org.edgexfoundry.test.data.ServiceData;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.data.domain.Sort;

@Category(RequiresNone.class)
public class ScheduleEventControllerTest {

  private static final String LIMIT_PROPERTY = "maxLimit";
  private static final int MAX_LIMIT = 100;

  private static final String TEST_ID = "123";
  private static final String TEST_ERR_MSG = "test message";

  @InjectMocks
  private ScheduleEventControllerImpl controller;

  @Mock
  private ScheduleEventRepository repos;

  @Mock
  private ScheduleRepository scheduleRepos;

  @Mock
  private ScheduleEventDao dao;

  @Mock
  private AddressableDao addressableDao;

  @Mock
  private CallbackExecutor callback;

  private ScheduleEvent event;

  @Before
  public void setup() throws Exception {
    MockitoAnnotations.initMocks(this);
    setControllerMAXLIMIT(MAX_LIMIT);
    event = ScheduleEventData.newTestInstance();
    event.setId(TEST_ID);
  }


  @Test
  public void testSchedule() {
    when(repos.findOne(TEST_ID)).thenReturn(event);
    assertEquals("Schedule event returned is not as expected", event,
        controller.scheduleEvent(TEST_ID));
  }

  @Test(expected = NotFoundException.class)
  public void testScheduleEventNotFound() {
    controller.scheduleEvent(TEST_ID);
  }

  @Test(expected = ServiceException.class)
  public void testScheduleEventException() {
    when(repos.findOne(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.scheduleEvent(TEST_ID);
  }

  @Test
  public void testScheduleEventss() {
    List<ScheduleEvent> evnts = new ArrayList<>();
    evnts.add(event);
    when(repos.findAll(any(Sort.class))).thenReturn(evnts);
    when(repos.count()).thenReturn(1L);
    List<ScheduleEvent> events = controller.scheduleEvents();
    assertEquals("Number of schedules events returned does not matched expected number", 1,
        events.size());
    assertEquals("Schedule events returned is not as expected", event, events.get(0));
  }

  @Test(expected = LimitExceededException.class)
  public void testScheduleEventEventsMaxLimit() {
    when(repos.count()).thenReturn(1000L);
    controller.scheduleEvents();
  }

  @Test(expected = ServiceException.class)
  public void testScheduleEventsException() {
    when(repos.findAll(any(Sort.class))).thenThrow(new RuntimeException(TEST_ERR_MSG));
    when(repos.count()).thenReturn(1L);
    controller.scheduleEvents();
  }

  @Test
  public void testScheduleEventForName() {
    when(repos.findByName(ScheduleEventData.TEST_SCHEDULE_EVENT_NAME)).thenReturn(event);
    assertEquals("Schedule event returned is not as expected", event,
        controller.scheduleEventForName(ScheduleEventData.TEST_SCHEDULE_EVENT_NAME));
  }

  @Test(expected = NotFoundException.class)
  public void testScheduleEventsForNameNotFound() {
    controller.scheduleEventForName(ScheduleEventData.TEST_SCHEDULE_EVENT_NAME);
  }

  @Test(expected = ServiceException.class)
  public void testScheduleEventsForNameException() {
    when(repos.findByName(ScheduleEventData.TEST_SCHEDULE_EVENT_NAME))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.scheduleEventForName(ScheduleEventData.TEST_SCHEDULE_EVENT_NAME);
  }

  @Test
  public void testScheduleEventsForAddressable() {
    Addressable addressable = AddressableData.newTestInstance();
    List<ScheduleEvent> evts = new ArrayList<>();
    evts.add(event);
    when(addressableDao.getById(TEST_ID)).thenReturn(addressable);
    when(repos.findByAddressable(addressable)).thenReturn(evts);
    List<ScheduleEvent> events = controller.scheduleEventsForAddressable(TEST_ID);
    assertEquals("Number of events returned does not matched expected number", 1, events.size());
    assertEquals("Service returned is not as expected", event, events.get(0));
  }

  @Test(expected = NotFoundException.class)
  public void testScheduleEventsForAddressableNoneFound() {
    when(addressableDao.getById(TEST_ID)).thenReturn(null);
    controller.scheduleEventsForAddressable(TEST_ID);
  }

  @Test(expected = ServiceException.class)
  public void testScheduleEventsForAddressableException() {
    when(addressableDao.getById(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.scheduleEventsForAddressable(TEST_ID);
  }

  @Test
  public void testScheduleEventsForAddressableByName() {
    Addressable addressable = AddressableData.newTestInstance();
    List<ScheduleEvent> evts = new ArrayList<>();
    evts.add(event);
    when(addressableDao.getByName(AddressableData.TEST_ADDR_NAME)).thenReturn(addressable);
    when(repos.findByAddressable(addressable)).thenReturn(evts);
    List<ScheduleEvent> events =
        controller.scheduleEventsForAddressableByName(AddressableData.TEST_ADDR_NAME);
    assertEquals("Number of events returned does not matched expected number", 1, events.size());
    assertEquals("Service returned is not as expected", event, events.get(0));
  }

  @Test(expected = NotFoundException.class)
  public void testScheduleEventForAddressableByNameNoneFound() {
    when(addressableDao.getByName(AddressableData.TEST_ADDR_NAME)).thenReturn(null);
    controller.scheduleEventsForAddressableByName(AddressableData.TEST_ADDR_NAME);
  }

  @Test(expected = ServiceException.class)
  public void testScheduleEventsForAddressableByNameException() {
    when(addressableDao.getByName(AddressableData.TEST_ADDR_NAME))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.scheduleEventsForAddressableByName(AddressableData.TEST_ADDR_NAME);
  }

  @Test
  public void testScheduleEventByService() {
    List<ScheduleEvent> evts = new ArrayList<>();
    evts.add(event);
    when(repos.findByService(ServiceData.TEST_SERVICE_NAME)).thenReturn(evts);
    List<ScheduleEvent> events =
        controller.scheduleEventsForServiceByName(ServiceData.TEST_SERVICE_NAME);
    assertEquals("Number of events returned does not matched expected number", 1, events.size());
    assertEquals("Event returned is not as expected", event, events.get(0));
  }

  @Test
  public void testScheduleEventByServiceWithNull() {
    List<ScheduleEvent> evts = new ArrayList<>();
    when(repos.findByService(null)).thenReturn(evts);
    List<ScheduleEvent> events = controller.scheduleEventsForServiceByName(null);
    assertEquals("Number of events returned does not matched expected number", 0, events.size());
  }

  @Test(expected = ServiceException.class)
  public void testScheduleEventByServiceException() {
    when(repos.findByService(ServiceData.TEST_SERVICE_NAME))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.scheduleEventsForServiceByName(ServiceData.TEST_SERVICE_NAME);
  }



  @Test
  public void testAdd() {
    Addressable addressable = AddressableData.newTestInstance();
    event.setAddressable(addressable);
    event.setSchedule(ScheduleData.TEST_SCHEDULE_NAME);
    when(scheduleRepos.findByName(ScheduleData.TEST_SCHEDULE_NAME))
        .thenReturn(ScheduleData.newTestInstance());
    when(addressableDao.getByIdOrName(addressable)).thenReturn(addressable);
    when(repos.save(event)).thenReturn(event);
    assertEquals("Schedule event returned was not the same as added", TEST_ID,
        controller.add(event));
  }

  @Test(expected = DataValidationException.class)
  public void testAddWithNull() {
    controller.add(null);
  }

  @Test(expected = DataValidationException.class)
  public void testAddWithDuplicatKey() {
    Addressable addressable = AddressableData.newTestInstance();
    event.setAddressable(addressable);
    event.setSchedule(ScheduleData.TEST_SCHEDULE_NAME);
    when(scheduleRepos.findByName(ScheduleData.TEST_SCHEDULE_NAME))
        .thenReturn(ScheduleData.newTestInstance());
    when(addressableDao.getByIdOrName(addressable)).thenReturn(addressable);
    when(repos.save(event)).thenThrow(new DuplicateKeyException(TEST_ERR_MSG));
    controller.add(event);
  }

  @Test(expected = ServiceException.class)
  public void testAddException() {
    Addressable addressable = AddressableData.newTestInstance();
    event.setAddressable(addressable);
    event.setSchedule(ScheduleData.TEST_SCHEDULE_NAME);
    when(scheduleRepos.findByName(ScheduleData.TEST_SCHEDULE_NAME))
        .thenReturn(ScheduleData.newTestInstance());
    when(addressableDao.getByIdOrName(addressable)).thenReturn(addressable);
    when(scheduleRepos.findByName(ScheduleData.TEST_SCHEDULE_NAME))
        .thenReturn(ScheduleData.newTestInstance());
    when(repos.save(event)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.add(event);
  }

  @Test(expected = DataValidationException.class)
  public void testAddWithNoSchedule() {
    event.setSchedule(null);
    controller.add(event);
  }

  @Test(expected = NotFoundException.class)
  public void testAddScheduleNotFound() {
    Addressable addressable = AddressableData.newTestInstance();
    event.setAddressable(addressable);
    event.setSchedule(ScheduleData.TEST_SCHEDULE_NAME);
    when(scheduleRepos.findByName(ScheduleData.TEST_SCHEDULE_NAME)).thenReturn(null);
    when(addressableDao.getByIdOrName(addressable)).thenReturn(addressable);
    when(repos.save(event)).thenReturn(event);
    controller.add(event);
  }

  @Test(expected = DataValidationException.class)
  public void testAddNoAssociatedAddressable() {
    Addressable addressable = AddressableData.newTestInstance();
    event.setAddressable(addressable);
    event.setSchedule(ScheduleData.TEST_SCHEDULE_NAME);
    when(scheduleRepos.findByName(ScheduleData.TEST_SCHEDULE_NAME))
        .thenReturn(ScheduleData.newTestInstance());
    when(addressableDao.getByIdOrName(addressable)).thenReturn(null);
    when(repos.save(event)).thenReturn(event);
    controller.add(event);
  }

  @Test
  public void testUpdate() {
    event.setSchedule(ScheduleData.TEST_SCHEDULE_NAME);
    when(scheduleRepos.findByName(ScheduleData.TEST_SCHEDULE_NAME))
        .thenReturn(ScheduleData.newTestInstance());
    when(dao.getByIdOrName(event)).thenReturn(event);
    assertTrue("Event was not updated", controller.update(event));
  }

  @Test(expected = ServiceException.class)
  public void testUpdateWithNull() {
    controller.update(null);
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateWithUnknownEvent() {
    when(dao.getByIdOrName(event)).thenReturn(null);
    controller.update(event);
  }

  @Test(expected = ServiceException.class)
  public void testUpdateException() {
    when(dao.getByIdOrName(event)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.update(event);
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateScheduleUpdateNotValide() {
    event.setSchedule(ScheduleData.TEST_SCHEDULE_NAME);
    when(scheduleRepos.findByName(ScheduleData.TEST_SCHEDULE_NAME)).thenReturn(null);
    when(dao.getByIdOrName(event)).thenReturn(event);
    controller.update(event);
  }

  @Test(expected = DataValidationException.class)
  public void testUpdateAssociatedToDeviceReport() {
    event.setSchedule(ScheduleData.TEST_SCHEDULE_NAME);
    when(dao.isScheduleEventAssociatedToDeviceReport(event)).thenReturn(true);
    when(scheduleRepos.findByName(ScheduleData.TEST_SCHEDULE_NAME))
        .thenReturn(ScheduleData.newTestInstance());
    when(dao.getByIdOrName(event)).thenReturn(event);
    controller.update(event);
  }

  @Test
  public void testDelete() {
    when(repos.findOne(TEST_ID)).thenReturn(event);
    assertTrue("Event was not deleted", controller.delete(TEST_ID));
  }

  @Test(expected = NotFoundException.class)
  public void testDeleteNotFound() {
    when(repos.findOne(TEST_ID)).thenReturn(null);
    controller.delete(TEST_ID);
  }

  @Test(expected = ServiceException.class)
  public void testDeleteException() {
    when(repos.findOne(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    assertTrue("Event was not deleted", controller.delete(TEST_ID));
  }


  @Test
  public void testDeleteByName() {
    when(repos.findByName(ScheduleEventData.TEST_SCHEDULE_EVENT_NAME)).thenReturn(event);
    assertTrue("Event was not deleted",
        controller.deleteByName(ScheduleEventData.TEST_SCHEDULE_EVENT_NAME));
  }

  @Test(expected = NotFoundException.class)
  public void testDeleteByNameNotFound() {
    when(repos.findByName(ScheduleEventData.TEST_SCHEDULE_EVENT_NAME)).thenReturn(null);
    controller.deleteByName(ScheduleEventData.TEST_SCHEDULE_EVENT_NAME);
  }

  @Test(expected = ServiceException.class)
  public void testDeleteByNameException() {
    when(repos.findByName(ScheduleEventData.TEST_SCHEDULE_EVENT_NAME))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    assertTrue("Event was not deleted",
        controller.deleteByName(ScheduleEventData.TEST_SCHEDULE_EVENT_NAME));
  }

  @Test(expected = DataValidationException.class)
  public void testDeleteWithAssociatedDeviceReport() {
    when(dao.isScheduleEventAssociatedToDeviceReport(event)).thenReturn(true);
    when(repos.findOne(TEST_ID)).thenReturn(event);
    controller.delete(TEST_ID);
  }

  private void setControllerMAXLIMIT(int newLimit) throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField(LIMIT_PROPERTY);
    temp.setAccessible(true);
    temp.set(controller, newLimit);
  }
}
