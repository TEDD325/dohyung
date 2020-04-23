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
import org.edgexfoundry.controller.impl.ScheduleControllerImpl;
import org.edgexfoundry.dao.ScheduleDao;
import org.edgexfoundry.dao.ScheduleRepository;
import org.edgexfoundry.domain.meta.Schedule;
import org.edgexfoundry.exception.controller.DataValidationException;
import org.edgexfoundry.exception.controller.LimitExceededException;
import org.edgexfoundry.exception.controller.NotFoundException;
import org.edgexfoundry.exception.controller.ServiceException;
import org.edgexfoundry.test.category.RequiresNone;
import org.edgexfoundry.test.data.ScheduleData;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.data.domain.Sort;

@Category(RequiresNone.class)
public class ScheduleControllerTest {

  private static final String LIMIT_PROPERTY = "maxLimit";
  private static final int MAX_LIMIT = 100;

  private static final String TEST_ID = "123";
  private static final String TEST_ERR_MSG = "test message";

  @InjectMocks
  private ScheduleControllerImpl controller;

  @Mock
  private ScheduleRepository repos;

  @Mock
  private ScheduleDao dao;

  @Mock
  private CallbackExecutor callback;

  private Schedule schedule;

  @Before
  public void setup() throws Exception {
    MockitoAnnotations.initMocks(this);
    setControllerMAXLIMIT(MAX_LIMIT);
    schedule = ScheduleData.newTestInstance();
    schedule.setId(TEST_ID);
  }

  @Test
  public void testSchedule() {
    when(repos.findOne(TEST_ID)).thenReturn(schedule);
    assertEquals("Schedule returned is not as expected", schedule, controller.schedule(TEST_ID));
  }

  @Test(expected = NotFoundException.class)
  public void testScheduleNotFound() {
    controller.schedule(TEST_ID);
  }

  @Test(expected = ServiceException.class)
  public void testScheduleException() {
    when(repos.findOne(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.schedule(TEST_ID);
  }

  @Test
  public void testSchedules() {
    List<Schedule> schs = new ArrayList<>();
    schs.add(schedule);
    when(repos.findAll(any(Sort.class))).thenReturn(schs);
    when(repos.count()).thenReturn(1L);
    List<Schedule> schedules = controller.schedules();
    assertEquals("Number of schedules returned does not matched expected number", 1,
        schedules.size());
    assertEquals("Schedules returned is not as expected", schedule, schedules.get(0));
  }

  @Test(expected = LimitExceededException.class)
  public void testDeviceServicesMaxLimit() {
    when(repos.count()).thenReturn(1000L);
    controller.schedules();
  }

  @Test(expected = ServiceException.class)
  public void testDeviceServicesException() {
    when(repos.findAll(any(Sort.class))).thenThrow(new RuntimeException(TEST_ERR_MSG));
    when(repos.count()).thenReturn(1L);
    controller.schedules();
  }

  @Test
  public void testScheduleForName() {
    when(repos.findByName(ScheduleData.TEST_SCHEDULE_NAME)).thenReturn(schedule);
    assertEquals("Schedule returned is not as expected", schedule,
        controller.scheduleForName(ScheduleData.TEST_SCHEDULE_NAME));
  }

  @Test(expected = NotFoundException.class)
  public void testSchedulesForNameNotFound() {
    controller.scheduleForName(ScheduleData.TEST_SCHEDULE_NAME);
  }

  @Test(expected = ServiceException.class)
  public void testSchedulesForNameException() {
    when(repos.findByName(ScheduleData.TEST_SCHEDULE_NAME))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.scheduleForName(ScheduleData.TEST_SCHEDULE_NAME);
  }

  @Test
  public void testAdd() {
    when(repos.save(schedule)).thenReturn(schedule);
    assertEquals("Schedule returned was not the same as added", TEST_ID, controller.add(schedule));
  }

  @Test(expected = ServiceException.class)
  public void testAddWithNull() {
    controller.add(null);
  }

  @Test(expected = DataValidationException.class)
  public void testAddWithDuplicatKey() {
    when(repos.save(schedule)).thenThrow(new DuplicateKeyException(TEST_ERR_MSG));
    controller.add(schedule);
  }

  @Test(expected = ServiceException.class)
  public void testAddException() {
    when(repos.save(schedule)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.add(schedule);
  }

  @Test(expected = DataValidationException.class)
  public void testAddWithBadCron() {
    schedule.setCron("foobar");
    when(repos.save(schedule)).thenReturn(schedule);
    assertEquals("Schedule returned was not the same as added", TEST_ID, controller.add(schedule));
  }

  @Test
  public void testUpdate() {
    when(dao.getByIdOrName(schedule)).thenReturn(schedule);
    assertTrue("Schedule was not updated", controller.update(schedule));
  }

  @Test(expected = ServiceException.class)
  public void testUpdateWithNull() {
    controller.update(null);
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateWithNoServiceID() {
    schedule.setId(null);
    when(dao.getByIdOrName(schedule)).thenReturn(null);
    controller.update(schedule);
  }

  @Test(expected = ServiceException.class)
  public void testUpdateException() {
    when(dao.getByIdOrName(schedule)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.update(schedule);
  }

  @Test
  public void testUpdateWithNoAssociation() {
    Schedule schedule2 = ScheduleData.newTestInstance();
    schedule.setName("foo");
    when(dao.getByIdOrName(schedule)).thenReturn(schedule2);
    controller.update(schedule);
  }

  @Test
  public void testUpdateWithAssociationAndNoViolation() {
    Schedule schedule2 = ScheduleData.newTestInstance();
    schedule.setName("foo");
    when(dao.isScheduleAssociatedToScheduleEvent(schedule2)).thenReturn(false);
    when(dao.getByIdOrName(schedule)).thenReturn(schedule2);
    controller.update(schedule);
  }

  @Test(expected = DataValidationException.class)
  public void testUpdateWithAssociationAndViolation() {
    Schedule schedule2 = ScheduleData.newTestInstance();
    schedule.setName("foo");
    when(dao.isScheduleAssociatedToScheduleEvent(schedule2)).thenReturn(true);
    when(dao.getByIdOrName(schedule)).thenReturn(schedule2);
    controller.update(schedule);
  }

  @Test
  public void testDelete() {
    when(repos.findOne(TEST_ID)).thenReturn(schedule);
    assertTrue("Schedule was not deleted", controller.delete(TEST_ID));
  }

  @Test(expected = NotFoundException.class)
  public void testDeleteNotFound() {
    when(repos.findOne(TEST_ID)).thenReturn(null);
    controller.delete(TEST_ID);
  }

  @Test(expected = ServiceException.class)
  public void testDeleteException() {
    when(repos.findOne(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    assertTrue("Schedule was not deleted", controller.delete(TEST_ID));
  }

  @Test
  public void testDeleteByName() {
    when(repos.findByName(ScheduleData.TEST_SCHEDULE_NAME)).thenReturn(schedule);
    assertTrue("Schedule was not deleted",
        controller.deleteByName(ScheduleData.TEST_SCHEDULE_NAME));
  }

  @Test(expected = NotFoundException.class)
  public void testDeleteByNameNotFound() {
    when(repos.findByName(ScheduleData.TEST_SCHEDULE_NAME)).thenReturn(null);
    controller.deleteByName(ScheduleData.TEST_SCHEDULE_NAME);
  }

  @Test(expected = ServiceException.class)
  public void testDeleteByNameException() {
    when(repos.findByName(ScheduleData.TEST_SCHEDULE_NAME))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    assertTrue("Schedule was not deleted",
        controller.deleteByName(ScheduleData.TEST_SCHEDULE_NAME));
  }

  @Test(expected = DataValidationException.class)
  public void testDeleteWtihAssociatedEvent() {
    when(dao.isScheduleAssociatedToScheduleEvent(schedule)).thenReturn(true);
    when(repos.findOne(TEST_ID)).thenReturn(schedule);
    controller.delete(TEST_ID);
  }

  private void setControllerMAXLIMIT(int newLimit) throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField(LIMIT_PROPERTY);
    temp.setAccessible(true);
    temp.set(controller, newLimit);
  }

}
